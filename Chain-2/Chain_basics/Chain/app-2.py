from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# basics

# LLM(chatmodel)

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

response = llm.invoke("What is the tallest building in the world?")
print(response.content)

# 1- Message
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

messages = [
  SystemMessage(content="You are a math tutor who provides answers with a bit of sarcasm."),
  HumanMessage(content="What is the square of 2?"),
]

response = llm.invoke(messages)
print(response.content)

# 2- PromptTemplate
from langchain_core.prompts import PromptTemplate

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

email_template = PromptTemplate.from_template(
  "Create an invitation email to the recipient that is {recipient_name}\
for an event that is {event_type}\
in a language that is {language}\
Mention the event location that is {event_location}\
and event date that is {event_date}.\
Also write few sentences about the event description that is {event_description}\
in style that is {style}."
)

details = {
  "recipient_name":"John",
  "event_type":"product launch",
  "language": "American english",
  "event_location":"Grand Ballroom, City Center Hotel",
  "event_date":"11 AM, January 15, 2024",
  "event_description":"an exciting unveiling of our latest GenAI product",
  "style":"enthusiastic tone"
}

prompt_value = email_template.invoke(details)
response = llm.invoke(prompt_value)
print(response.content)

# OutputParser
from langchain_core import PydanticOutputParser
from pydantic import BaseModel, Field

class Author(BaseModel):
    name: str = Field(description="The name of the author")
    number: int = Field(description="The number of books written by the author")
    books: list[str] = Field(description="The list of books they wrote")

pydantic_parser = PydanticOutputParser(pydantic_object=Author)

prompt_list = PromptTemplate.from_template(
    template = "Answer the question.\n{format_instructions}\n{question}",
    input_vairables = ["question"],
    partial_variables = {"format_instructions": pydantic_parser.get_format_instructions()},
)

prompt_value = prompt_list.invoke({"question": "Generate the books written by Dan Brown"})

response = llm.invoke(prompt_value)
returned_object = pydantic_parser.parse(response.content)
