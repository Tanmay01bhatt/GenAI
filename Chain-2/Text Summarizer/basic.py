from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant that summaries text in a clear and concise manner."),
    ("user","Summarize the following text :"
     "\n Text : {text}")
])

parser = StrOutputParser()

chain = prompt | llm |parser

text = input("Enter text to summarize: ")

response = chain.invoke({"text": text})
print("Summary:",response)