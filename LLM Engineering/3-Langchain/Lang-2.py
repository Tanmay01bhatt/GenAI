#OUTPUT PARSERS

# DateTimeOutputParser

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import DatetimeOutputParser

llm = ChatGroq(model="llama3-8b-8192")

parser_dateTime = DatetimeOutputParser()
prompt_dateTime = PromptTemplate.from_template(
    template = "Answer the question.\n{format_instructions}\n{question}",
    input_vairables = ["question"],
    partial_variables = {"format_instructions": parser_dateTime.get_format_instructions()}
)

prompt_value = prompt_dateTime.invoke({"question": "When was the iPhone released"})
response = llm.invoke(prompt_value)
print(response.content)

returned_object = parser_dateTime.parse(response.content)
print(type(returned_object))

# CSV List Parser
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser

llm = ChatGroq(model="llama3-8b-8192")

parser_list = CommaSeparatedListOutputParser()
prompt_list = PromptTemplate.from_template(
    template = "Answer the question.\n{format_instructions}\n{question}",
    input_vairables = ["question"],
    partial_variables = {"format_instructions": parser_list.get_format_instructions()},
)

prompt_value = prompt_list.invoke({"question": "List 4 chocolate brands"})
response = llm.invoke(prompt_value)
print(response.content)

returned_object = parser_list.parse(response.content)
print(type(returned_object))

# Pydantic Parser
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

llm = ChatGroq(model="llama3-8b-8192")

class Author(BaseModel):
    name: str = Field(description="The name of the author")
    number: int = Field(description="The number of books written by the author")
    books: list[str] = Field(description="The list of books they wrote")

output_parser = PydanticOutputParser(pydantic_object=Author)

prompt_list = PromptTemplate.from_template(
    template = "Answer the question.\n{format_instructions}\n{question}",
    input_vairables = ["question"],
    partial_variables = {"format_instructions": output_parser.get_format_instructions()},
)

prompt_value = prompt_list.invoke({"question": "Generate the books written by Dan Brown"})
response = llm.invoke(prompt_value)

returned_object = output_parser.parse(response.content)
print(f"{returned_object.name} wrote {returned_object.number} books.")
print(returned_object.books)

# Getting a Strucutered Output Without Using Parsers
#.with_structured_output() 

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

llm = ChatGroq(model="llama3-8b-8192")

class Author(BaseModel):
    name: str = Field(description="The name of the author")
    number: int = Field(description="The number of books written by the author")
    books: list[str] = Field(description="The list of books they wrote")

structured_llm = llm.with_structured_output(Author)
returned_object = structured_llm.invoke("Generate the books written by Dan Brown")

print(f"{returned_object.name} wrote {returned_object.number} books.")
print(returned_object.books)
