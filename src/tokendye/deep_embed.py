"""DeepEmbed-style dyeing: per-layer, per-label channel scaling.

An alternative to the original "post-embed" method (which dyed token
embeddings once at the embedding layer), this module applies a
per-token-role element-wise scaling to the hidden states of every
transformer layer:

    hidden[layer] = hidden[layer] * (1 + delta[label][layer])

Each (label, layer) pair owns a trainable vector `delta` of length d_model,
initialized to zeros, so a pretrained model starts from the identity map and
fine-tuning only nudges values away from 0. `delta` is stored as an fp32
master weight and cast to the hidden dtype on every forward, so small
updates are not lost to bf16 rounding.

Parameter cost (4 labels, 32 layers, 4096 dims, ~7B-class model):
    4 * 32 * 4096 = 524,288 params ~= 1 MB in FP16.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = ["DeepEmbedConfig", "DeepEmbedController", "DeepEmbedDye", "install_deep_embed"]


@dataclass
class DeepEmbedConfig:
    """Dye configuration.

    Attributes:
        labels: Role label names, e.g. ["system", "user", "agent", "response"].
        num_layers: Number of transformer layers to dye.
        d_model: Hidden size of the transformer.
        dtype: torch dtype name, e.g. "float32" / "bfloat16".
    """

    labels: Sequence[str]
    num_layers: int
    d_model: int
    dtype: str = "float32"

    def build(self) -> DeepEmbedDye:
        """Build a dye module from this config."""
        return DeepEmbedDye.from_config(self)


class DeepEmbedDye(nn.Module):
    """Per-layer, per-label channel scaling vectors (scale = 1 + delta).

    `delta` has shape (num_labels, num_layers, d_model) and is initialized to
    zeros; the effective scale is `1 + delta`. `delta` is the fp32 master
    weight, cast to the hidden dtype on each forward.
    """

    def __init__(
        self,
        labels: Sequence[str],
        num_layers: int,
        d_model: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be unique")
        self.labels = list(labels)
        self.num_layers = int(num_layers)
        self.d_model = int(d_model)
        self.label_to_id = {name: i for i, name in enumerate(self.labels)}
        self.delta = nn.Parameter(
            torch.zeros(len(self.labels), self.num_layers, self.d_model, dtype=dtype)
        )

    @classmethod
    def from_config(cls, config: DeepEmbedConfig) -> DeepEmbedDye:
        """Build a dye module from a config."""
        return cls(
            config.labels,
            config.num_layers,
            config.d_model,
            dtype=getattr(torch, config.dtype),
        )

    @property
    def num_parameters(self) -> int:
        """Total dye parameter count (= labels x layers x hidden size)."""
        return self.delta.numel()

    @property
    def scale(self) -> Tensor:
        """Effective scaling (fp32 view): scale = 1 + delta."""
        return 1 + self.delta

    def apply_layer(self, hidden: Tensor, labels: Tensor, layer_idx: int) -> Tensor:
        """Scale one layer's hidden states element-wise by per-token labels.

        Args:
            hidden: (batch, seq, d_model) hidden states of the current layer.
            labels: (batch, seq) label id per token; -1 leaves the token undyed.
            layer_idx: Current layer index (0-based).

        Returns:
            Scaled hidden states with the same shape.
        """
        labels = labels.to(device=hidden.device)
        layer_delta = self.delta[:, layer_idx, :].to(dtype=hidden.dtype)  # (num_labels, d_model)
        valid = (labels >= 0) & (labels < self.delta.shape[0])
        safe = labels.clamp(0, self.delta.shape[0] - 1)
        token_delta = layer_delta[safe]  # (batch, seq, d_model)
        token_delta = torch.where(
            valid.unsqueeze(-1),
            token_delta,
            torch.zeros_like(token_delta),
        )
        return hidden * (1 + token_delta)

    def forward(self, hidden: Tensor, labels: Tensor, layer_idx: int) -> Tensor:
        return self.apply_layer(hidden, labels, layer_idx)


class _DyeMaskHolder:
    """Batch label mask shared by all layer hooks."""

    def __init__(self, labels: Tensor | None = None) -> None:
        self.labels = labels


class DeepEmbedController:
    """Handle to the installed per-layer dye hooks.

    Usage:
        controller = install_deep_embed(model, dye)
        controller.set_labels(mask)     # set the label mask before each forward
        ...
        controller.remove()             # remove all hooks
    """

    def __init__(
        self,
        dye: DeepEmbedDye,
        layer_modules: Sequence[nn.Module],
        holder: _DyeMaskHolder,
        hook_handles: Sequence,
    ) -> None:
        self.dye = dye
        self.layers = list(layer_modules)
        self._holder = holder
        self._hook_handles = list(hook_handles)

    @property
    def labels(self) -> Tensor | None:
        return self._holder.labels

    def set_labels(self, labels: Tensor | None) -> None:
        """Set the label mask for the current batch, shape (batch, seq)."""
        self._holder.labels = labels

    def clear_labels(self) -> None:
        """Clear the label mask (equivalent to no dye)."""
        self._holder.labels = None

    def remove(self) -> None:
        """Remove all dye hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._holder.labels = None


_KNOWN_LAYER_PATHS = (
    "model.layers",
    "model.model.layers",
    "language_model.model.layers",
    "model.language_model.model.layers",
    "model.language_model.layers",
    "transformer.h",
    "model.decoder.layers",
    "model.encoder.layer",
)


def _get_attr_path(obj: nn.Module, path: str) -> nn.Module | None:
    cur: nn.Module | None = obj
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _discover_layers(model: nn.Module, num_layers: int) -> list[nn.Module]:
    """Locate the transformer layer list (common HF layouts + fallback)."""
    for path in _KNOWN_LAYER_PATHS:
        mod = _get_attr_path(model, path)
        if isinstance(mod, (nn.ModuleList, nn.Sequential)) and len(mod) == num_layers:
            return list(mod)

    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.ModuleList, nn.Sequential)) or len(mod) == 0:
            continue
        first = mod[0]
        if any(hasattr(first, attr) for attr in ("self_attn", "attn", "attention")):
            candidates.append((name, mod))

    if not candidates:
        raise ValueError(
            "Could not find a transformer layer list; pass layers= explicitly "
            "(e.g. model.model.layers)"
        )
    exact = [(name, mod) for name, mod in candidates if len(mod) == num_layers]
    if len(exact) == 1:
        return list(exact[0][1])
    pool = exact or candidates
    raise ValueError(
        "Could not uniquely determine the layer list to dye; pass layers= "
        "explicitly. Candidates: " + ", ".join(f"{name} ({len(mod)} layers)" for name, mod in pool)
    )


def _make_layer_hook(dye: DeepEmbedDye, holder: _DyeMaskHolder, layer_idx: int):
    def hook(module, args, output):
        labels = holder.labels
        if labels is None:
            return output
        if isinstance(output, (tuple, list)):
            head, *rest = output
            return (dye.apply_layer(head, labels, layer_idx), *rest)
        return dye.apply_layer(output, labels, layer_idx)

    return hook


def install_deep_embed(
    model: nn.Module,
    dye: DeepEmbedDye,
    layers: Sequence[nn.Module] | None = None,
) -> DeepEmbedController:
    """Register per-layer dye hooks on the model and return a controller.

    Args:
        model: An HF-style model (common layer layouts are auto-detected) or
            any nn.Module.
        dye: A DeepEmbedDye instance.
        layers: Layer modules to dye; by default auto-discovered.

    Returns:
        A DeepEmbedController. Use set_labels(mask) to supply per-batch labels
        and remove() to uninstall all hooks.
    """
    layer_modules = list(layers) if layers is not None else _discover_layers(model, dye.num_layers)
    if len(layer_modules) != dye.num_layers:
        raise ValueError(
            f"Layer count mismatch: dye has {dye.num_layers} layers, "
            f"got {len(layer_modules)} layer modules"
        )

    holder = _DyeMaskHolder()
    hook_handles = []
    for layer_idx, layer_module in enumerate(layer_modules):
        handle = layer_module.register_forward_hook(_make_layer_hook(dye, holder, layer_idx))
        hook_handles.append(handle)
    return DeepEmbedController(dye, layer_modules, holder, hook_handles)
