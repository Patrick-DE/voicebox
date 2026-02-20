"""
Custom model management - store and retrieve user-defined HuggingFace model configurations.
"""

import json
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Optional

from . import config


def _get_custom_models_path() -> Path:
    """Get path to custom models config file."""
    return config.get_data_dir() / "custom_models.json"


def _load_custom_models() -> List[Dict]:
    """Load custom models from storage."""
    path = _get_custom_models_path()
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def _save_custom_models(models: List[Dict]) -> None:
    """Save custom models to storage."""
    path = _get_custom_models_path()
    with open(path, "w") as f:
        json.dump(models, f, indent=2)


def parse_hf_url(url: str) -> str:
    """
    Parse a HuggingFace URL or repo ID to extract the repo ID.

    Supported formats:
        "https://huggingface.co/hexgrad/Kokoro-82M"  -> "hexgrad/Kokoro-82M"
        "hexgrad/Kokoro-82M"                          -> "hexgrad/Kokoro-82M"

    Raises:
        ValueError: If the input is a Space URL or otherwise invalid.
    """
    url = url.strip()

    # Reject HuggingFace Spaces URLs - these are demos, not model repos
    if re.match(r"https?://huggingface\.co/spaces/", url):
        raise ValueError(
            "HuggingFace Spaces URLs are not supported. "
            "Please use the model repository URL instead "
            "(e.g., 'https://huggingface.co/hexgrad/Kokoro-82M')."
        )

    # Parse full HuggingFace model URL
    hf_match = re.match(r"https?://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", url)
    if hf_match:
        return hf_match.group(1)

    # Accept bare repo IDs (owner/model-name)
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", url):
        return url

    raise ValueError(
        f"Invalid HuggingFace URL or repo ID: '{url}'. "
        "Expected 'owner/model-name' or 'https://huggingface.co/owner/model-name'."
    )


def repo_id_to_model_name(repo_id: str) -> str:
    """Convert a HuggingFace repo ID to a safe internal model name."""
    sanitized = re.sub(r"[^A-Za-z0-9-]", "-", repo_id.replace("/", "-"))
    return f"custom-{sanitized}"


def add_custom_model(hf_url: str, display_name: Optional[str] = None) -> Dict:
    """
    Register a custom model by HuggingFace URL or repo ID.

    Returns the model config dict (existing entry if already registered).
    Raises ValueError for invalid URLs.
    """
    repo_id = parse_hf_url(hf_url)
    model_name = repo_id_to_model_name(repo_id)

    existing = _load_custom_models()
    for m in existing:
        if m["hf_repo_id"] == repo_id:
            return m  # Already registered

    if display_name is None:
        display_name = repo_id.split("/")[-1]

    model_config = {
        "model_name": model_name,
        "display_name": display_name,
        "hf_repo_id": repo_id,
    }

    existing.append(model_config)
    _save_custom_models(existing)
    return model_config


def list_custom_models() -> List[Dict]:
    """Return all registered custom models."""
    return _load_custom_models()


def get_custom_model(model_name: str) -> Optional[Dict]:
    """Return a custom model config by internal model name, or None if not found."""
    for m in _load_custom_models():
        if m["model_name"] == model_name:
            return m
    return None


def remove_custom_model(model_name: str) -> bool:
    """
    Remove a custom model registration.

    Returns True if removed, False if not found.
    Does NOT delete the HuggingFace cache on disk.
    """
    models = _load_custom_models()
    new_models = [m for m in models if m["model_name"] != model_name]
    if len(new_models) == len(models):
        return False
    _save_custom_models(new_models)
    return True


def download_custom_model_sync(model_name: str, repo_id: str) -> None:
    """
    Synchronously download a custom model from HuggingFace with progress tracking.

    Intended to be called via asyncio.to_thread so it doesn't block the event loop.
    """
    from .utils.progress import get_progress_manager
    from .utils.tasks import get_task_manager
    from .utils.hf_progress import HFProgressTracker, create_hf_progress_callback
    from huggingface_hub import snapshot_download

    progress_manager = get_progress_manager()
    task_manager = get_task_manager()

    progress_callback = create_hf_progress_callback(model_name, progress_manager)
    tracker = HFProgressTracker(progress_callback, filter_non_downloads=False)

    tracker_context = tracker.patch_download()
    tracker_context.__enter__()
    try:
        snapshot_download(repo_id=repo_id)
        progress_manager.mark_complete(model_name)
        task_manager.complete_download(model_name)
    except Exception as e:
        progress_manager.mark_error(model_name, str(e))
        task_manager.error_download(model_name, str(e))
        raise
    finally:
        tracker_context.__exit__(None, None, None)
