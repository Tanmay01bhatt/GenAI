#Tool calling : llm deciding if or when to use a tool .

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

@tool
def calculate_discount(price:float,dis:float)->float:
    '''
    Calculates the final price after applying a discount.

    Args:
        price (float): The original price of the item.
        dis (float): The discount percentage 

    Returns:
        float: The final price after the discount is applied.
    '''
    disc = price*(dis/100)
    amt = price - disc
    return amt

# bind the tools
llm_with_tools =  llm.bind_tools([calculate_discount])

# decision making

q1 = llm_with_tools.invoke("hello world")
q2 = llm_with_tools.invoke("what is the price of an item that costs $1000 after 12% discount?")

print(q1.content,'\n')
print(q2.tool_calls)  # attribtes of a tool_calls

#name: The name of the tool to use.
#args: The input arguments for the tool.
#id: An identifier of the specific call.

arg = q2.tool_calls[0]['args']
result = calculate_discount.invoke(arg)
print(result)