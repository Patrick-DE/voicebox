"""
Pluggable model adapters for different TTS architectures.

Each adapter wraps a specific TTS library behind a common interface,
allowing the PyTorchTTSBackend to dispatch to the correct one based
on the model type.
"""

from typing import Optional, Tuple
import numpy as np


class ModelAdapter:
    """
    Base interface for custom model adapters.

    Subclasses wrap a specific TTS library (XTTS, Chatterbox, etc.)
    and provide a uniform generate() that accepts reference audio
    for voice cloning.
    """

    model_type: str = "unknown"

    def load(self, model_id: str, device: str) -> None:
        """Load the model. `model_id` is the HF repo ID."""
        raise NotImplementedError

    def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
    ) -> dict:
        """
        Create a voice prompt dict from reference audio.

        For adapters that handle cloning internally during generate(),
        this just stores the paths. For adapters that pre-process the
        prompt, this does that work.
        """
        # Default: just store the paths for later use
        return {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }

    def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text with voice cloning.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with at least 'ref_audio' and 'ref_text'
            language: Language code (e.g. 'en', 'zh', 'de')
            seed: Optional random seed
            instruct: Optional instruction (ignored by most adapters)

        Returns:
            Tuple of (audio_numpy_array, sample_rate)
        """
        raise NotImplementedError

    def unload(self) -> None:
        """Release model resources."""
        pass


# ===========================================================================
# XTTS v2 Adapter (Coqui TTS)
# ===========================================================================

class XTTSAdapter(ModelAdapter):
    """
    Adapter for XTTS v2 via the `TTS` (coqui-ai) library.

    Install: pip install TTS
    Supports 17 languages, zero-shot voice cloning from ~6s reference audio.
    """

    model_type = "xtts"

    # XTTS language code mapping (ISO 639-1 → XTTS)
    LANG_MAP = {
        "en": "en", "zh": "zh-cn", "ja": "ja", "ko": "ko",
        "de": "de", "fr": "fr", "ru": "ru", "pt": "pt",
        "es": "es", "it": "it", "pl": "pl", "tr": "tr",
        "nl": "nl", "cs": "cs", "ar": "ar", "hu": "hu",
        "hi": "hi",
    }

    def __init__(self):
        self._tts = None

    def load(self, model_id: str, device: str) -> None:
        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError(
                "XTTS v2 requires the 'TTS' package. "
                "Install it with: pip install TTS"
            )

        print(f"Loading XTTS v2 model on {device}...")
        # model_id is typically "tts_models/multilingual/multi-dataset/xtts_v2"
        # but we also accept the HF repo name; Coqui TTS resolves it
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("XTTS v2 loaded successfully")

    def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        if self._tts is None:
            raise RuntimeError("XTTS model not loaded")

        ref_audio = voice_prompt.get("ref_audio", "")
        lang = self.LANG_MAP.get(language, "en")

        # Generate with voice cloning
        wav = self._tts.tts(
            text=text,
            speaker_wav=ref_audio,
            language=lang,
        )

        audio = np.array(wav, dtype=np.float32)
        # XTTS v2 outputs at 24000 Hz
        return audio, 24000

    def unload(self) -> None:
        if self._tts is not None:
            del self._tts
            self._tts = None


# ===========================================================================
# Chatterbox Adapter (Resemble AI)
# ===========================================================================

class ChatterboxAdapter(ModelAdapter):
    """
    Adapter for Chatterbox TTS by Resemble AI.

    Install: pip install chatterbox-tts
    MIT license, zero-shot voice cloning from ~5s reference audio.
    """

    model_type = "chatterbox"

    def __init__(self):
        self._model = None
        self._device = "cpu"

    def load(self, model_id: str, device: str) -> None:
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError:
            raise ImportError(
                "Chatterbox requires the 'chatterbox-tts' package. "
                "Install it with: pip install chatterbox-tts"
            )

        print(f"Loading Chatterbox TTS model on {device}...")
        self._device = device
        self._model = ChatterboxTTS.from_pretrained(device=device)
        print("Chatterbox TTS loaded successfully")

    def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        import torch

        if self._model is None:
            raise RuntimeError("Chatterbox model not loaded")

        ref_audio = voice_prompt.get("ref_audio", "")

        # Chatterbox generates a torch tensor
        wav = self._model.generate(
            text=text,
            audio_prompt_path=ref_audio,
        )

        # Convert to numpy
        if isinstance(wav, torch.Tensor):
            audio = wav.squeeze().cpu().numpy().astype(np.float32)
        else:
            audio = np.array(wav, dtype=np.float32)

        # Chatterbox outputs at 24000 Hz
        return audio, 24000

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None


# ===========================================================================
# Registry
# ===========================================================================

# Known HuggingFace repo IDs → adapter type
KNOWN_MODELS = {
    "coqui/XTTS-v2": "xtts",
    "coqui/xtts-v2": "xtts",
    # Chatterbox doesn't have a single canonical HF repo,
    # it's loaded via from_pretrained()
    "resemble-ai/chatterbox": "chatterbox",
    "ResembleAI/chatterbox": "chatterbox",
}

# Adapter class registry
ADAPTER_CLASSES = {
    "xtts": XTTSAdapter,
    "chatterbox": ChatterboxAdapter,
}


def detect_model_type(hf_repo_id: str) -> str:
    """
    Detect the model type from a HuggingFace repo ID.

    Returns 'xtts', 'chatterbox', or 'qwen' (default).
    """
    # Check known models
    if hf_repo_id in KNOWN_MODELS:
        return KNOWN_MODELS[hf_repo_id]

    # Heuristic: check repo name for keywords
    lower = hf_repo_id.lower()
    if "xtts" in lower:
        return "xtts"
    if "chatterbox" in lower:
        return "chatterbox"

    # Default to Qwen (the built-in backend)
    return "qwen"


def create_adapter(model_type: str) -> Optional[ModelAdapter]:
    """
    Create an adapter instance for the given model type.

    Returns None for 'qwen' (handled by the existing PyTorch backend directly).
    """
    if model_type == "qwen":
        return None

    adapter_class = ADAPTER_CLASSES.get(model_type)
    if adapter_class is None:
        print(f"Warning: Unknown model type '{model_type}', falling back to Qwen")
        return None

    return adapter_class()
