from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

# os.environ["GROQ_API_KEY"] = "key"

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer clearly.\nQuestion: {question}"
)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

query = input("Ask something: ")
response = chain.invoke({"question": query})

print("\nResponse:", response)