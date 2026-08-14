"""Qwen3.5 example configuration."""

from pathlib import Path

LABELS = ["system", "user", "agent", "response"]
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "Qwen3.5-4B")
AGENT_ID = LABELS.index("agent")
RESPONSE_ID = LABELS.index("response")
