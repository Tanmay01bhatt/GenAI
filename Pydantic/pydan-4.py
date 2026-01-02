#optional feild
from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    id: int    # Feilds
    name: str
    email: Optional[str] = None
    sal : Optional[int] = None

user = User(id=1, name="Tanmay")
print(user)

# we can also give a list 

from typing import List

class User2(BaseModel):
    id:int
    name:List[str]

user2 = User2(id=2, name=["Tanmay","Divy",'Devansh'])
print(user2)