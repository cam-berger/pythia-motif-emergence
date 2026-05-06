"""Generic path-patching primitive (Goldowsky-Dill 2023 protocol).

Implements the frozen-paths variant: for a given sender head and a set of
receiver heads, run a forward pass where the sender's output is replaced with
its corrupt-prompt value (at a specified position) and all attention/MLP
outputs at intermediate layers are pinned to their clean-prompt values. The
only path along which the corruption can reach a receiver is the direct
residual-stream addition.

The pinning rule (for sender at layer L_s and receivers at layers >= L_r):
  - Sender's z at (L_s, sender_head, sender_position): replace with corrupt
    cache value (other heads at L_s are not pinned — they read identical input
    to clean and produce identical output naturally).
  - Sender's MLP at L_s: pin to clean (the residual stream at MLP input is
    perturbed, so without pinning the MLP would compute differently).
  - All layers in (L_s, max(receiver_layers)) exclusive: pin both attn_out
    and mlp_out to clean.
  - Receiver layers and beyond: no pin. Receiver attention runs naturally on
    the patched residual stream; we capture its hook_pattern for each receiver.

This is detector-agnostic. The S-inhibition-specific bookkeeping (corruption
prompt construction, S2/IO position lookup, scalar reduction across NMs) lives
in `src/detectors/s_inhibition.py`. Path-patching for other detectors that fit
the same shape (single sender → multiple receivers, attention-pattern readout)
can reuse this primitive directly.

Reference: Goldowsky-Dill et al. 2023 ("Localizing Model Behavior with Path
Patching"); same protocol used in Wang 2023's IOI circuit identification and
Conmy et al. ACDC 2023.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from transformer_lens import ActivationCache, HookedTransformer


@dataclass(frozen=True)
class PathPatchScreen:
    """Output of `path_patch_screen` across many senders.

    `clean_patterns[layer]` has shape `(B, n_heads, T, T)` and is the
    receiver-layer attention pattern under the unperturbed clean run; this is
    sender-independent and is computed once.

    `patched_patterns[(sender_layer, sender_head)][layer]` has shape
    `(B, n_heads, T, T)` and is the receiver-layer attention pattern under the
    path-patched run with that sender's z replaced at `sender_position`.
    """

    clean_patterns: dict[int, torch.Tensor]
    patched_patterns: dict[tuple[int, int], dict[int, torch.Tensor]]


def _make_pin_hook(clean_value: torch.Tensor) -> Callable:
    def fn(activation, hook):
        return clean_value
    return fn


def _make_sender_z_hook(
    head: int, position: int, corrupt_z: torch.Tensor
) -> Callable:
    def fn(z, hook):
        z[:, position, head, :] = corrupt_z[:, position, head, :]
        return z
    return fn


def _make_pattern_capture(layer: int, store: dict[int, torch.Tensor]) -> Callable:
    def fn(pattern, hook):
        store[layer] = pattern.detach().clone()
    return fn


def _build_freeze_hooks(
    *,
    sender_layer: int,
    sender_head: int,
    sender_position: int,
    max_receiver_layer: int,
    clean_cache: ActivationCache,
    corrupt_cache: ActivationCache,
) -> list[tuple[str, Callable]]:
    """Build the list of (hook_name, hook_fn) for one path-patched forward."""
    hooks: list[tuple[str, Callable]] = []

    sender_z_name = f"blocks.{sender_layer}.attn.hook_z"
    corrupt_z = corrupt_cache[sender_z_name].detach().clone()
    hooks.append(
        (sender_z_name, _make_sender_z_hook(sender_head, sender_position, corrupt_z))
    )

    sender_mlp_name = f"blocks.{sender_layer}.hook_mlp_out"
    hooks.append(
        (sender_mlp_name, _make_pin_hook(clean_cache[sender_mlp_name].detach().clone()))
    )

    for layer in range(sender_layer + 1, max_receiver_layer):
        attn_out_name = f"blocks.{layer}.hook_attn_out"
        mlp_out_name = f"blocks.{layer}.hook_mlp_out"
        hooks.append(
            (attn_out_name, _make_pin_hook(clean_cache[attn_out_name].detach().clone()))
        )
        hooks.append(
            (mlp_out_name, _make_pin_hook(clean_cache[mlp_out_name].detach().clone()))
        )

    return hooks


def path_patch_one_sender(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    *,
    clean_cache: ActivationCache,
    corrupt_cache: ActivationCache,
    sender_layer: int,
    sender_head: int,
    sender_position: int,
    receiver_layers: list[int],
) -> dict[int, torch.Tensor]:
    """Run one path-patched forward pass and capture receiver attention.

    Returns a dict mapping each receiver layer to that layer's attention
    pattern tensor of shape `(B, n_heads, T, T)`. The dict contains one entry
    per layer in `receiver_layers` (deduplicated).
    """
    if sender_layer >= max(receiver_layers):
        raise ValueError(
            f"sender at layer {sender_layer} must be strictly before "
            f"max(receiver_layers)={max(receiver_layers)}; upstream patching "
            f"is not supported by this primitive."
        )

    max_receiver_layer = max(receiver_layers)
    freeze_hooks = _build_freeze_hooks(
        sender_layer=sender_layer,
        sender_head=sender_head,
        sender_position=sender_position,
        max_receiver_layer=max_receiver_layer,
        clean_cache=clean_cache,
        corrupt_cache=corrupt_cache,
    )

    captured: dict[int, torch.Tensor] = {}
    pattern_hooks = [
        (f"blocks.{l}.attn.hook_pattern", _make_pattern_capture(l, captured))
        for l in sorted(set(receiver_layers))
    ]

    with torch.no_grad():
        model.run_with_hooks(
            clean_tokens,
            fwd_hooks=freeze_hooks + pattern_hooks,
            return_type=None,
        )

    return captured


def _required_cache_names(
    model: HookedTransformer, sender_layers: set[int], receiver_layers: set[int]
) -> set[str]:
    """Cache names we actually need for clean and corrupt runs.

    For the clean cache: hook_attn_out + hook_mlp_out at every layer between
    the earliest sender and the latest receiver (used for pinning), plus
    hook_pattern at receiver layers (used to expose clean comparison values).
    For the corrupt cache: only hook_z at sender layers (the only thing the
    sender hook reads from corrupt).
    """
    if not sender_layers or not receiver_layers:
        return set()
    earliest = min(sender_layers)
    latest = max(receiver_layers)
    layers_to_pin = range(earliest, latest)  # exclusive of latest
    names: set[str] = set()
    for layer in layers_to_pin:
        names.add(f"blocks.{layer}.hook_attn_out")
        names.add(f"blocks.{layer}.hook_mlp_out")
    for layer in receiver_layers:
        names.add(f"blocks.{layer}.attn.hook_pattern")
    for layer in sender_layers:
        names.add(f"blocks.{layer}.attn.hook_z")
    return names


def path_patch_screen(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    *,
    senders: list[tuple[int, int]],
    receivers: list[tuple[int, int]],
    sender_position: int = -1,
) -> PathPatchScreen:
    """Path-patch every sender against the union of receiver layers.

    Computes clean and corrupt caches once, then iterates over senders running
    one path-patched forward each. Returns clean attention patterns at the
    receiver layers (sender-independent) and patched attention patterns
    indexed by sender.

    The `corrupt_tokens` must have identical shape to `clean_tokens`. Mixed
    sequence lengths within a batch are not supported; the caller is
    responsible for grouping prompts by length.

    Args:
        model: TransformerLens model.
        clean_tokens: `(B, T)` clean prompt tokens.
        corrupt_tokens: `(B, T)` ABC-corrupted prompt tokens.
        senders: list of `(layer, head)` pairs to iterate over as senders.
        receivers: list of `(layer, head)` pairs identifying the receivers
            whose attention patterns we want at each receiver layer. Note
            that the primitive captures the *whole layer's* attention pattern,
            not just the listed heads' rows; the caller selects per-head
            slices afterward.
        sender_position: position at which to replace the sender's z. -1 = END.

    Returns: `PathPatchScreen` with `clean_patterns` and `patched_patterns`.
    """
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError(
            f"clean_tokens shape {tuple(clean_tokens.shape)} != "
            f"corrupt_tokens shape {tuple(corrupt_tokens.shape)}"
        )
    if not senders:
        raise ValueError("senders must be non-empty")
    if not receivers:
        raise ValueError("receivers must be non-empty")

    sender_layers = {l for l, _ in senders}
    receiver_layers = {l for l, _ in receivers}
    cache_names = _required_cache_names(model, sender_layers, receiver_layers)
    name_filter = lambda name: name in cache_names  # noqa: E731

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(
            clean_tokens, names_filter=name_filter, return_type=None
        )
        _, corrupt_cache = model.run_with_cache(
            corrupt_tokens, names_filter=name_filter, return_type=None
        )

    receiver_layers_sorted = sorted(receiver_layers)
    clean_patterns = {
        layer: clean_cache[f"blocks.{layer}.attn.hook_pattern"].detach().clone()
        for layer in receiver_layers_sorted
    }

    patched_patterns: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
    for sl, sh in senders:
        patched_patterns[(sl, sh)] = path_patch_one_sender(
            model,
            clean_tokens,
            clean_cache=clean_cache,
            corrupt_cache=corrupt_cache,
            sender_layer=sl,
            sender_head=sh,
            sender_position=sender_position,
            receiver_layers=receiver_layers_sorted,
        )

    return PathPatchScreen(
        clean_patterns=clean_patterns,
        patched_patterns=patched_patterns,
    )
