from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
text =input("Enter text to summarize: ")
documents = text_splitter.create_documents([text])

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant that summaries text in a clear and concise manner"),
    ("human", "Summarize the text in 5 concise bullet points.\n\nText :{text}")
])
parser = StrOutputParser()
chain = prompt | llm | parser

chunk_summaries = []
for doc in documents:
    summary = chain.invoke({"text": doc.page_content})
    chunk_summaries.append(summary)

# Combine summaries
final_summary = "\n\n".join(chunk_summaries)

print("\nFINAL SUMMARY:\n")
print(final_summary)