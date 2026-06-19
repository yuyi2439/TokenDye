from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from .label import DyeLabel

if TYPE_CHECKING:
    from os import PathLike


class DyeConfig(BaseModel):
    model_name: str
    labels: list[DyeLabel]
    rank: int
    d_model: int
    dtype: str

    def save(self, path: Optional["PathLike | str"] = None, indent: int = 2):
        path = path or f"DyeConfig_{self.model_name}.json"
        p = Path(path)
        p.write_text(self.model_dump_json(indent=indent))

    @classmethod
    def load(cls, path: "PathLike | str"):
        p = Path(path)
        return cls.model_validate_json(p.read_text())
