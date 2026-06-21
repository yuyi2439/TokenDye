from pydantic import BaseModel


class DyeLabel(BaseModel):
    name: str
    id: int
