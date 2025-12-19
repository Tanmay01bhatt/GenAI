
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
load_dotenv()

loader = PyPDFLoader('attention.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, #Maximum number of characters per chunk
    chunk_overlap=100 #characters from the previous chunk are repeated in the next chunk to maintain context
)

chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 3})

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant. Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""
)

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough() #forwards the input unchanged to the next step in the chain.(so that the question remains unchanged).
    }
    | prompt
    | llm
    | StrOutputParser()
)

query = input("Ask a question: ")
response = rag_chain.invoke(query)

print("\nAnswer:", response)