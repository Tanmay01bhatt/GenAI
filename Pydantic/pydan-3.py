# Custom Logic Validation using @validator decorator

from pydantic import BaseModel, field_validator , model_validator

class User(BaseModel):
    age : int
    name :str

    @field_validator('age')
    def check_age(cls, value):
        if value < 18:
            raise ValueError("Age must be at least 18")
        return value
    
user = User(age=15, name="Tanmay")
print(user)


# model validation (can access multiple fields)
class Delivery(BaseModel):
    pickup: int
    drop: int

    @model_validator(mode='before')
    @classmethod 
    def fix_input(cls, data):
        print("Before validator sees raw input:", data)
        # Let's swap them if they are reversed
        if int(data['drop']) < int(data['pickup']):
            data['pickup'], data['drop'] = data['drop'], data['pickup']
        return data
    

order1 = Delivery(pickup=15, drop = 13)
print("After model validation:", order1.model_dump())
