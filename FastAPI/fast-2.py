from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn 

class Item(BaseModel):
    text: str = None
    is_done: bool = False


app = FastAPI()

items = []

@app.get("/")# home
def index():
    return {"message": " Bye bye world!"}

@app.post("/items")
def add_items(item:Item):
    items.append(item)
    return items


@app.get("/items", response_model=list[Item]) # get items by slicing
def list_items(limit: int = 10):
    return items[0:limit]

@app.get("/items/{item_id}", response_model=Item) # get item by id
def get_item(item_id: int) -> Item:
    if item_id < len(items):
        return items[item_id]
    else:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)
