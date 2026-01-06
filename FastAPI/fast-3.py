from fastapi import FastAPI, HTTPException ,status,Path
from pydantic import BaseModel
import uvicorn 
from typing import Optional

app = FastAPI()

@app.get('/') # www.example.com/
def root():
    return {"message":"Home Page"}

# database
users_db = {
    1: {"name":"Alice","age":30},
    2: {"name":"Bob","age":25},
    3: {"name":"Charlie","age":35}
}

class User(BaseModel):
    name : str
    age : int

class UpdateUser(BaseModel):
    name : Optional[str] =None
    age : Optional[int] = None

# GET 
# Fetch data from db
@app.get('/user/{userid}') # www.example.com/user/123
def get_user(userid:int):
    if userid not in users_db:
        raise HTTPException(status_code=404,detail="User not found")
    return users_db[userid]

# POST 
#create user
@app.post('/user/{userid}',status_code=status.HTTP_201_CREATED) # www.example.com/user/4
def create_user(userid:int,user:User):
    if userid in users_db:
        raise HTTPException(status_code=400,detail="User already exists")
    users_db[userid] = user.model_dump() #pyd to dict
    return users_db[userid]

#PUT
# update user

@app.put('/user/{userid}')
def update_user(userid:int,user:UpdateUser):
    if userid not in users_db:
        raise HTTPException(status_code=400,detail="User not here")
    
    current_user = users_db[userid]

    if user.name is not None:
        current_user['name'] = user.name
    if user.age is not None:
        current_user['age'] = user.age

    return current_user
    
# DELETE
@app.delete('/user/{userid}')
def delete_user(userid:int):    
    if userid not in users_db:
        raise HTTPException(status_code=404,detail="User not found")
    del_user = users_db.pop(userid)
    return {"message":"User has been deleted","User":del_user}

# SEARCH
@app.get("/user/search/") # www.example.com/user/search/?name=Alice
def search_user_by_name(name:Optional[str]=None):
    if not name:
        return {"message":"Name is Required"}
    
    for user in users_db.values():
        if user['name'] == name:
            return user
    raise HTTPException(status_code=404,detail="User not found")