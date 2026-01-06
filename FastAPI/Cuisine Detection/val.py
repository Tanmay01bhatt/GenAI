from pydantic import BaseModel

class TextVal(BaseModel):
    text: str
    