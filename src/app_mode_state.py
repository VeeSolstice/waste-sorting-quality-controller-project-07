"""Small state helpers that keep the two Streamlit workflows isolated."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


CONTROLLED_MODE = "Controlled background (default)"
EXPERIMENTAL_MODE = "Experimental learned TACO detector"
APP_MODES = (CONTROLLED_MODE, EXPERIMENTAL_MODE)

CONTROLLED_RESULT_KEYS = (
    "scene_analysis",
    "scene_analysis_key",
    "scene_upload_name",
)
EXPERIMENTAL_RESULT_KEYS = (
    "detector_analysis",
    "detector_analysis_key",
    "detector_upload_name",
)


def isolate_mode_state(state: MutableMapping[str, Any], active_mode: str) -> bool:
    """Clear both workflows' results when the user changes analysis mode."""
    previous_mode = state.get("active_analysis_mode")
    changed = previous_mode is not None and previous_mode != active_mode
    if changed:
        for key in (*CONTROLLED_RESULT_KEYS, *EXPERIMENTAL_RESULT_KEYS):
            state.pop(key, None)
    state["active_analysis_mode"] = active_mode
    return changed
