#Data Validation using in-built data types 

from pydantic import BaseModel,EmailStr

class User(BaseModel):
    id: int    # Feilds
    name: str
    email: str

user = User(id=1, name=18, email="tanma@gmail.com")

print(user.name) # error

# validation for an invalid email
class User2(BaseModel):
    id: int    # Feilds
    name: str
    email: EmailStr

user2 = User2(id=2, name="Tanmay", email="bhatt")
print(user2.email) # error