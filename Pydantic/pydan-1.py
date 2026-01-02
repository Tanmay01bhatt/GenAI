from pydantic import BaseModel

class User(BaseModel):
    id: int    # Feilds
    name: str
    email: str

user = User(id=1, name="Tanmay", email="tanma@gmail.com")
print(user)
print(user.name)

# @dataclass decorator is not great for data validation