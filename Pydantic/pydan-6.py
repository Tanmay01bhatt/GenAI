# Json Serialization : converting a Pydantic model into JSON-compatible data

from pydantic import BaseModel

class Product(BaseModel):
    id: int
    name: str
    tags: list[str]
    in_stock: bool

product = Product( id=1,name="Notebook",tags=["office", "paper"], in_stock=True)

# serialize to python dict
print(product.model_dump())

# or serialize to JSON string
print(product.model_dump())