"""MPS-compatibility helpers.

The brief flags several MPS gotchas (PROJECT_BRIEF.md §7): bf16 unreliable,
fused=False in Adam, kernel deaths under default tutorials. This module
centralizes the workarounds so detector/sweep code can stay clean.
"""

from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    """Return MPS if available, else CPU. CUDA is not supported by this project."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def assert_mps_fallback_enabled() -> None:
    """Verify PYTORCH_ENABLE_MPS_FALLBACK is set in the environment.

    TransformerLens hits ops that MPS doesn't implement; without the fallback
    flag, those ops raise instead of running on CPU.
    """
    flag = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    if flag != "1":
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK is not set to '1'. "
            "Add `export PYTORCH_ENABLE_MPS_FALLBACK=1` to your shell rc, "
            "or set it in the current process before importing torch."
        )


def safe_dtype(prefer: str = "fp32") -> torch.dtype:
    """Return a dtype known to work on MPS.

    bf16 is unreliable on MPS as of 2026; fp16 works but loses precision for
    detector arithmetic. Default fp32 unless caller has measured otherwise.
    """
    if prefer == "fp16":
        return torch.float16
    if prefer == "fp32":
        return torch.float32
    raise ValueError(f"unsupported MPS dtype preference: {prefer!r}")
