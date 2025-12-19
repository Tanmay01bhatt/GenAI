# CHAT_MODEL
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-8b-8192")

response = llm.invoke("What is the tallest building in the world?")
print(response.content)

# Messages

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatGroq(model="llama3-8b-8192")

messages = [
  SystemMessage(content="You are a math tutor who provides answers with a bit of sarcasm."),
  HumanMessage(content="What is the square of 2?"),
]

response = llm.invoke(messages)
print(response.content)

# Prompt Templates
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

llm = ChatGroq(model="llama3-8b-8192")

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