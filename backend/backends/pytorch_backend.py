"""
PyTorch backend implementation for TTS and STT.
"""

from typing import Optional, List, Tuple
import asyncio
import re
import torch
import numpy as np
from pathlib import Path

from . import TTSBackend, STTBackend
from ..utils.cache import get_cache_key, get_cached_voice_prompt, cache_voice_prompt
from ..utils.audio import normalize_audio, load_audio
from ..utils.progress import get_progress_manager
from ..utils.hf_progress import HFProgressTracker, create_hf_progress_callback
from ..utils.tasks import get_task_manager

LANGUAGE_CODE_TO_NAME = {
    "zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean",
    "de": "german", "fr": "french", "ru": "russian", "pt": "portuguese",
    "es": "spanish", "it": "italian",
}


class PyTorchTTSBackend:
    """PyTorch-based TTS backend using Qwen3-TTS."""
    
    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self.device = self._get_device()
        self._current_model_size = None
        self._model_type = "qwen"  # "qwen", "xtts", "chatterbox", etc.
        self._adapter = None  # ModelAdapter instance for non-Qwen models
    
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS can have issues, use CPU for stability
            return "cpu"
        return "cpu"
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    @staticmethod
    def _model_progress_name(model_size: str) -> str:
        """Return a safe progress-tracking name for the given model identifier."""
        if model_size in ("1.7B", "0.6B"):
            return f"qwen-tts-{model_size}"
        return "custom-" + re.sub(r"[^A-Za-z0-9-]", "-", model_size.replace("/", "-"))
    
    def _get_model_path(self, model_size: str) -> str:
        """
        Get the HuggingFace Hub model ID.
        
        Args:
            model_size: Model size ('1.7B', '0.6B') or a raw HF repo ID for custom models
            
        Returns:
            HuggingFace Hub model ID
        """
        hf_model_map = {
            "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        }
        
        if model_size in hf_model_map:
            return hf_model_map[model_size]
        
        # For custom models, model_size is already the HF repo ID (e.g. "hexgrad/Kokoro-82M")
        return model_size
    
    def _is_model_cached(self, model_size: str) -> bool:
        """
        Check if the model is already cached locally AND fully downloaded.
        
        Args:
            model_size: Model size to check
            
        Returns:
            True if model is fully cached, False if missing or incomplete
        """
        try:
            from huggingface_hub import constants as hf_constants
            model_path = self._get_model_path(model_size)
            repo_cache = Path(hf_constants.HF_HUB_CACHE) / ("models--" + model_path.replace("/", "--"))
            
            if not repo_cache.exists():
                return False
            
            # Check for .incomplete files - if any exist, download is still in progress
            blobs_dir = repo_cache / "blobs"
            if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
                print(f"[_is_model_cached] Found .incomplete files for {model_size}, treating as not cached")
                return False
            
            # Check that actual model weight files exist in snapshots
            snapshots_dir = repo_cache / "snapshots"
            if snapshots_dir.exists():
                has_weights = (
                    any(snapshots_dir.rglob("*.safetensors")) or
                    any(snapshots_dir.rglob("*.bin"))
                )
                if not has_weights:
                    print(f"[_is_model_cached] No model weights found for {model_size}, treating as not cached")
                    return False
            
            return True
        except Exception as e:
            print(f"[_is_model_cached] Error checking cache for {model_size}: {e}")
            return False
    
    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the TTS model with automatic downloading from HuggingFace Hub.
        
        Args:
            model_size: Model size to load (1.7B or 0.6B)
        """
        if model_size is None:
            model_size = self.model_size
            
        # If already loaded with correct size, return
        if self.model is not None and self._current_model_size == model_size:
            return
        
        # Unload existing model if different size requested
        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()
        
        # Run blocking load in thread pool
        await asyncio.to_thread(self._load_model_sync, model_size)
    
    # Alias for compatibility
    load_model = load_model_async
    
    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        try:
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            model_name = self._model_progress_name(model_size)

            # Check if model is already cached
            is_cached = self._is_model_cached(model_size)

            # Determine model type for custom models
            model_type = "qwen"  # default
            if model_size not in ("1.7B", "0.6B"):
                # Custom model — detect type from repo ID or stored config
                from .model_adapters import detect_model_type
                model_type = detect_model_type(model_size)

            # Set up progress callback and tracker
            progress_callback = create_hf_progress_callback(model_name, progress_manager)
            tracker = HFProgressTracker(progress_callback, filter_non_downloads=is_cached)

            # Patch tqdm BEFORE importing any TTS library
            tracker_context = tracker.patch_download()
            tracker_context.__enter__()

            # Get model path (local or HuggingFace Hub ID)
            model_path = self._get_model_path(model_size)

            print(f"Loading TTS model {model_size} (type={model_type}) on {self.device}...")

            # Only track download progress if model is NOT cached
            if not is_cached:
                task_manager.start_download(model_name)
                progress_manager.update_progress(
                    model_name=model_name,
                    current=0,
                    total=0,
                    filename="Connecting to HuggingFace...",
                    status="downloading",
                )

            try:
                if model_type != "qwen":
                    # Load via adapter (XTTS, Chatterbox, etc.)
                    from .model_adapters import create_adapter
                    adapter = create_adapter(model_type)
                    if adapter is None:
                        raise ValueError(
                            f"No adapter available for model type '{model_type}'. "
                            f"This model architecture is not supported."
                        )
                    adapter.load(model_path, self.device)
                    self._adapter = adapter
                    self._model_type = model_type
                    self.model = adapter  # store adapter as 'model' so is_loaded() works
                else:
                    # Load Qwen3 TTS (original path)
                    from qwen_tts import Qwen3TTSModel
                    self.model = Qwen3TTSModel.from_pretrained(
                        model_path,
                        device_map=self.device,
                        torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
                    )
                    self._adapter = None
                    self._model_type = "qwen"
            finally:
                tracker_context.__exit__(None, None, None)
            
            # Only mark download as complete if we were tracking it
            if not is_cached:
                progress_manager.mark_complete(model_name)
                task_manager.complete_download(model_name)
            
            self._current_model_size = model_size
            self.model_size = model_size
            
            print(f"TTS model {model_size} loaded successfully (type={self._model_type})")
            
        except ImportError as e:
            print(f"Error: Required package not found: {e}")
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            model_name = self._model_progress_name(model_size)
            progress_manager.mark_error(model_name, str(e))
            task_manager.error_download(model_name, str(e))
            raise
        except Exception as e:
            print(f"Error loading TTS model: {e}")
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            model_name = self._model_progress_name(model_size)
            progress_manager.mark_error(model_name, str(e))
            task_manager.error_download(model_name, str(e))
            raise
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self._adapter is not None:
            self._adapter.unload()
            self._adapter = None
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None
            self._model_type = "qwen"
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("TTS model unloaded")
    
    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.
        
        Args:
            audio_path: Path to reference audio file
            reference_text: Transcript of reference audio
            use_cache: Whether to use cached prompt if available
            
        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        await self.load_model_async(None)
        
        # For adapter-based models, voice prompt is just the ref audio path
        # (adapters handle cloning internally during generate())
        if self._adapter is not None:
            prompt = self._adapter.create_voice_prompt(str(audio_path), reference_text)
            return prompt, False
        
        # Qwen3 path: create voice clone prompt
        # Check cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cached_prompt = get_cached_voice_prompt(cache_key)
            if cached_prompt is not None:
                if isinstance(cached_prompt, dict):
                    return cached_prompt, True
                elif isinstance(cached_prompt, torch.Tensor):
                    return {"prompt": cached_prompt}, True
        
        def _create_prompt_sync():
            """Run synchronous voice prompt creation in thread pool."""
            return self.model.create_voice_clone_prompt(
                ref_audio=str(audio_path),
                ref_text=reference_text,
                x_vector_only_mode=False,
            )
        
        voice_prompt_items = await asyncio.to_thread(_create_prompt_sync)
        
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cache_voice_prompt(cache_key, voice_prompt_items)
        
        return voice_prompt_items, False
    
    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        """
        Combine multiple reference samples for better quality.
        
        Args:
            audio_paths: List of audio file paths
            reference_texts: List of reference texts
            
        Returns:
            Tuple of (combined_audio, combined_text)
        """
        combined_audio = []
        
        for audio_path in audio_paths:
            audio, sr = load_audio(audio_path)
            audio = normalize_audio(audio)
            combined_audio.append(audio)
        
        # Concatenate audio
        mixed = np.concatenate(combined_audio)
        mixed = normalize_audio(mixed)
        
        # Combine texts
        combined_text = " ".join(reference_texts)
        
        return mixed, combined_text
    
    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text using voice prompt.

        Args:
            text: Text to synthesize
            voice_prompt: Voice prompt dictionary from create_voice_prompt
            language: Language code (en or zh)
            seed: Random seed for reproducibility
            instruct: Natural language instruction for speech delivery control

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Load model
        await self.load_model_async(None)

        # Dispatch to adapter for non-Qwen models
        if self._adapter is not None:
            def _adapter_generate():
                if seed is not None:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)
                return self._adapter.generate(
                    text=text,
                    voice_prompt=voice_prompt,
                    language=language,
                    seed=seed,
                    instruct=instruct,
                )
            audio, sample_rate = await asyncio.to_thread(_adapter_generate)
            return audio, sample_rate

        # Qwen3 path
        def _generate_sync():
            """Run synchronous generation in thread pool."""
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            wavs, sample_rate = self.model.generate_voice_clone(
                text=text,
                voice_clone_prompt=voice_prompt,
                language=LANGUAGE_CODE_TO_NAME.get(language, "auto"),
                instruct=instruct,
            )
            return wavs[0], sample_rate

        audio, sample_rate = await asyncio.to_thread(_generate_sync)
        return audio, sample_rate


class PyTorchSTTBackend:
    """PyTorch-based STT backend using Whisper."""
    
    def __init__(self, model_size: str = "base"):
        self.model = None
        self.processor = None
        self.model_size = model_size
        self.device = self._get_device()
    
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS support for Whisper
            return "cpu"  # Use CPU for stability
        return "cpu"
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def _is_model_cached(self, model_size: str) -> bool:
        """
        Check if the Whisper model is already cached locally AND fully downloaded.
        
        Args:
            model_size: Model size to check
            
        Returns:
            True if model is fully cached, False if missing or incomplete
        """
        try:
            from huggingface_hub import constants as hf_constants
            model_name = f"openai/whisper-{model_size}"
            repo_cache = Path(hf_constants.HF_HUB_CACHE) / ("models--" + model_name.replace("/", "--"))
            
            if not repo_cache.exists():
                return False
            
            # Check for .incomplete files - if any exist, download is still in progress
            blobs_dir = repo_cache / "blobs"
            if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
                print(f"[_is_model_cached] Found .incomplete files for whisper-{model_size}, treating as not cached")
                return False
            
            # Check that actual model weight files exist in snapshots
            snapshots_dir = repo_cache / "snapshots"
            if snapshots_dir.exists():
                has_weights = (
                    any(snapshots_dir.rglob("*.safetensors")) or
                    any(snapshots_dir.rglob("*.bin"))
                )
                if not has_weights:
                    print(f"[_is_model_cached] No model weights found for whisper-{model_size}, treating as not cached")
                    return False
            
            return True
        except Exception as e:
            print(f"[_is_model_cached] Error checking cache for whisper-{model_size}: {e}")
            return False
    
    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        print(f"[DEBUG] load_model_async called with size: {model_size}")
        if model_size is None:
            model_size = self.model_size

        print(f"[DEBUG] Model already loaded? {self.model is not None}, current size: {self.model_size}, requested: {model_size}")
        if self.model is not None and self.model_size == model_size:
            print(f"[DEBUG] Early return - model already loaded")
            return

        print(f"[DEBUG] Calling asyncio.to_thread for _load_model_sync")
        # Run blocking load in thread pool
        await asyncio.to_thread(self._load_model_sync, model_size)
        print(f"[DEBUG] asyncio.to_thread completed")
    
    # Alias for compatibility
    load_model = load_model_async
    
    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        print(f"[DEBUG] _load_model_sync called for Whisper {model_size}")
        try:
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            progress_model_name = f"whisper-{model_size}"

            # Check if model is already cached
            is_cached = self._is_model_cached(model_size)

            # Set up progress callback and tracker
            # If cached: filter out non-download progress
            # If not cached: report all progress (we're actually downloading)
            progress_callback = create_hf_progress_callback(progress_model_name, progress_manager)
            tracker = HFProgressTracker(progress_callback, filter_non_downloads=is_cached)

            # Patch tqdm BEFORE importing transformers
            print("[DEBUG] Starting tqdm patch BEFORE transformers import")
            tracker_context = tracker.patch_download()
            tracker_context.__enter__()
            print("[DEBUG] tqdm patched, now importing transformers")

            # Import transformers
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            model_name = f"openai/whisper-{model_size}"
            print(f"[DEBUG] Model name: {model_name}")

            print(f"Loading Whisper model {model_size} on {self.device}...")

            # Only track download progress if model is NOT cached
            if not is_cached:
                # Start tracking download task
                task_manager.start_download(progress_model_name)

                # Initialize progress state so SSE endpoint has initial data to send
                progress_manager.update_progress(
                    model_name=progress_model_name,
                    current=0,
                    total=0,  # Will be updated once actual total is known
                    filename="Connecting to HuggingFace...",
                    status="downloading",
                )

            # Load models (tqdm is patched, but filters out non-download progress)
            try:
                self.processor = WhisperProcessor.from_pretrained(model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            finally:
                # Exit the patch context
                tracker_context.__exit__(None, None, None)
            
            # Only mark download as complete if we were tracking it
            if not is_cached:
                progress_manager.mark_complete(progress_model_name)
                task_manager.complete_download(progress_model_name)
            
            self.model.to(self.device)
            self.model_size = model_size
            
            print(f"Whisper model {model_size} loaded successfully")
            
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            progress_model_name = f"whisper-{model_size}"
            progress_manager.mark_error(progress_model_name, str(e))
            task_manager.error_download(progress_model_name, str(e))
            raise
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("Whisper model unloaded")
    
    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            language: Optional language hint (en or zh)
            
        Returns:
            Transcribed text
        """
        await self.load_model_async(None)
        
        def _transcribe_sync():
            """Run synchronous transcription in thread pool."""
            # Load audio
            audio, sr = load_audio(audio_path, sample_rate=16000)
            
            # Process audio
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Set language if provided
            forced_decoder_ids = None
            if language:
                # Support all languages from frontend: en, zh, ja, ko, de, fr, ru, pt, es, it
                # Whisper supports these and many more
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language,
                    task="transcribe",
                )
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs["input_features"],
                    forced_decoder_ids=forced_decoder_ids,
                )
            
            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0]
            
            return transcription.strip()
        
        # Run blocking transcription in thread pool
        return await asyncio.to_thread(_transcribe_sync)
