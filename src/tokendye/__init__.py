"""TokenDye: a lightweight way to dye LLM hidden states with role labels."""

from .deep_embed import (
    DeepEmbedConfig,
    DeepEmbedController,
    DeepEmbedDye,
    install_deep_embed,
)

__all__ = [
    "DeepEmbedConfig",
    "DeepEmbedController",
    "DeepEmbedDye",
    "install_deep_embed",
]
