"""
Settings management for voicebox backend.

Stores user settings like HuggingFace tokens in a JSON file within the data directory.
"""

import json
from pathlib import Path
from typing import Optional

from . import config


def _get_settings_path() -> Path:
    """Get path to settings file."""
    return config.get_data_dir() / "settings.json"


def _load_settings() -> dict:
    """Load settings from disk."""
    path = _get_settings_path()
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_settings(settings: dict) -> None:
    """Save settings to disk."""
    path = _get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)


def get_hf_token() -> Optional[str]:
    """Get the stored HuggingFace token, or None if not set."""
    settings = _load_settings()
    return settings.get("hf_token")


def set_hf_token(token: str) -> None:
    """Store a HuggingFace token and apply it to the environment."""
    import os
    settings = _load_settings()
    settings["hf_token"] = token
    _save_settings(settings)
    # Also set as environment variable so huggingface_hub picks it up
    os.environ["HF_TOKEN"] = token
    # Login to huggingface_hub for the current session
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
    except Exception as e:
        print(f"Warning: Could not login to HuggingFace Hub: {e}")


def clear_hf_token() -> None:
    """Remove the stored HuggingFace token."""
    import os
    settings = _load_settings()
    settings.pop("hf_token", None)
    _save_settings(settings)
    os.environ.pop("HF_TOKEN", None)


def apply_saved_token() -> None:
    """Apply the saved HF token on startup (if any)."""
    token = get_hf_token()
    if token:
        import os
        os.environ["HF_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print("HuggingFace token loaded from settings")
        except Exception as e:
            print(f"Warning: Could not apply saved HF token: {e}")
