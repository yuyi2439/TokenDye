"""TokenDye: a lightweight way to dye LLM hidden states with role labels."""

from .module import (
    Dye,
    DyeConfig,
    DyeController,
    install_dye,
)

__all__ = [
    "Dye",
    "DyeConfig",
    "DyeController",
    "install_dye",
]
