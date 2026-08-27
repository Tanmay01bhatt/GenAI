#A tool is a function enriched with metadata.(must contain a docstring)
from langchain_core.tools import tool
# use @tool decorator
@tool
def calculate_discount(price: float, discount_percentage: float) -> float:
    """
    Calculates the final price after applying a discount.

    Args:
        price (float): The original price of the item.
        discount_percentage (float): The discount percentage (e.g., 20 for 20%).

    """
    if not (0 <= discount_percentage <= 100):
        raise ValueError("Discount percentage must be between 0 and 100")

    discount_amount = price * (discount_percentage / 100)
    final_price = price - discount_amount
    return final_price

# 3 tool attributes:
print(calculate_discount.name)
print(calculate_discount.description)
print(calculate_discount.args)

print(calculate_discount.invoke({"price":100, "discount_percentage": 15}))

# 2nd way is using BaseTool Interface = requires explicit declaration instead of docstring and lengthy