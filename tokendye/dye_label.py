from pydantic import BaseModel
from typing_extensions import deprecated


class DyeLabel(BaseModel):
    name: str
    id: int


@deprecated("")
class DyeLabelManager(dict[str, DyeLabel]):
    """继承自 `dict[str, DyeLabel]`
    
    """
    def __init__(self, labelset: set[str]):
        labels = enumerate(labelset)
        for id, name in labels:
            label = DyeLabel(name=name, id=id)
            self[name] = label
