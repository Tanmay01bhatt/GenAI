from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_db(path):
  loader = PyPDFLoader(file_path)
  chunks = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
  docs = text_splitter.split_documents(chunks)
  db = FAISS.from_documents(docs, embeddings)
  return db

def rag_chain(db):
  retriever = db.as_retriever(search_kwargs={"k": 5})
  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
  prompt = ChatPromptTemplate.from_messages([
    ('system','You are medical assistant'
     "Analyze the document carefully and aswer only using the information from the document."
      " If the information is not available, respond with 'I don't know.'"
      "Do not make assumptions."
      "You must only respond in bullet points"),
      ("user", "Answer the following question based on the provided context:\n{context}"
      "\n\nQuestion: {question}")])
  chain = {
    "context":retriever,
    "question":RunnablePassthrough()
  }| prompt | llm | StrOutputParser()

  return chain

file_path = 'DIABETES.pdf'
db = create_db(file_path)
query = input("Ask a query: ")
chain = rag_chain(db)
response = chain.invoke(query)

print(response)

