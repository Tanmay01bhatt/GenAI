from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
text =input("Enter text to summarize: ")
documents = text_splitter.create_documents([text])

llm = ChatGroq( model="llama-3.1-8b-instant",temperature=0)


pre_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant that summaries text in a clear and concise manner"),
    ("human", "Summarize the text in concise bullet points.\n\nText :{text}")
])
parser = StrOutputParser()
pre_chain = pre_prompt | llm | parser

chunk_summaries = []
for doc in documents:
     summary = pre_chain.invoke({"text": doc.page_content})
     chunk_summaries.append(summary)

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant that summaries text in a clear and concise manner"),
    ("human", "You are given multiple summaries of parts of a document."
     "Combine into a single summary of 5 concise bullet points."
     "\n\nText :{text}")
])

chain = prompt | llm | parser

final_summary = chain.invoke({"text": "\n\n".join(chunk_summaries)})

print("\nFINAL SUMMARY:\n")
print(final_summary)
