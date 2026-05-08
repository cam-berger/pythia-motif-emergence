"""Checkpoint-aware Pythia loader.

Pythia's HuggingFace repos expose every training checkpoint as a git revision
(e.g., revision='step143000'). TransformerLens' HookedTransformer.from_pretrained
accepts these via the `checkpoint_value` argument.

Usage:
    from src.utils.pythia_loader import load_pythia, prefetch_pythia
    prefetch_pythia("410m", step=143000)  # one-time download
    model = load_pythia("410m", step=143000)

Known issue: as of transformers + huggingface_hub combinations seen in
2026-05, an unauthenticated `from_pretrained` call can fail with
`httpx.LocalProtocolError: Illegal header value b'Bearer '` because
something in the transformers chain constructs an empty bearer token.
Workaround: pre-download via `prefetch_pythia` (which uses the working
huggingface_hub.snapshot_download path), then load with HF_HUB_OFFLINE=1
to bypass the failing auth header construction. `load_pythia` will set
HF_HUB_OFFLINE=1 in-process if the model is already cached locally.
"""

from __future__ import annotations

import os
from typing import Literal

import torch
from huggingface_hub import snapshot_download, try_to_load_from_cache
from transformer_lens import HookedTransformer

from src.utils.mps_compat import get_device, safe_dtype

PythiaSize = Literal["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]

_MODEL_NAME_TEMPLATE = "EleutherAI/pythia-{size}-deduped"


def pythia_repo(size: PythiaSize) -> str:
    """Return the HF repo name for a given Pythia size (deduped variant)."""
    return _MODEL_NAME_TEMPLATE.format(size=size)


def revision_for_step(step: int) -> str:
    """Pythia checkpoints are tagged as `step{N}` on HF (e.g., step143000)."""
    return f"step{step}"


_SNAPSHOT_PATTERNS = [
    "*.json",
    "*.bin",
    "*.safetensors",
    "tokenizer*",
    "special_tokens*",
    "merges*",
    "vocab*",
]


def prefetch_pythia(size: PythiaSize, step: int) -> str:
    """Download a Pythia checkpoint to the local HF cache.

    Uses huggingface_hub.snapshot_download, which works around the empty-bearer
    auth bug seen via transformers' from_pretrained path. Returns the local
    snapshot directory path.
    """
    return snapshot_download(
        repo_id=pythia_repo(size),
        revision=revision_for_step(step),
        allow_patterns=_SNAPSHOT_PATTERNS,
    )


def _is_cached(size: PythiaSize, step: int) -> bool:
    """Return True if config.json for (size, step) is already in HF cache."""
    cached = try_to_load_from_cache(
        repo_id=pythia_repo(size),
        filename="config.json",
        revision=revision_for_step(step),
    )
    return cached is not None and cached is not False


def load_pythia(
    size: PythiaSize,
    step: int = 143000,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> HookedTransformer:
    """Load a Pythia model at a specific training step into TransformerLens.

    Args:
        size: Pythia size string ("70m", "160m", "410m", ...).
        step: training step. Defaults to 143000 (final checkpoint).
        device: torch device. Defaults to MPS if available, else CPU.
        dtype: numerical dtype. Defaults to fp32 (MPS-safe).

    Returns:
        A TransformerLens HookedTransformer ready for hooking.
    """
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = safe_dtype("fp32")

    model_name = pythia_repo(size)
    revision = revision_for_step(step)

    # Workaround for the empty-Bearer auth bug: if the checkpoint is already
    # in the local HF cache, force offline mode for this load to avoid the
    # transformers->huggingface_hub network path that triggers the bug.
    prev_offline = os.environ.get("HF_HUB_OFFLINE")
    if _is_cached(size, step):
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            checkpoint_value=step,
            device=device,
            dtype=dtype,
        )
    finally:
        if prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_offline
    # Document the checkpoint we actually loaded (for downstream provenance).
    model.cfg.tokenizer_name = model_name
    model.cfg.checkpoint_label = revision
    # Sanity check: TransformerLens occasionally loads a stub tokenizer
    # (vocab_size=2) for some Pythia sizes (observed at 2.8b where the
    # checkpoint stores only pytorch_model.bin and not safetensors). If
    # detected, replace with a directly-loaded AutoTokenizer from the same
    # revision. This is a safety override; for sizes where TransformerLens
    # loads correctly (1b, 410m, etc.) it's a no-op (vocab_size > 100).
    tok = getattr(model, "tokenizer", None)
    if tok is not None and getattr(tok, "vocab_size", 0) < 100:
        from transformers import AutoTokenizer
        good_tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model.tokenizer = good_tok
    return model
