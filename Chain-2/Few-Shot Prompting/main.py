from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

loader = PyPDFLoader("Tree of Thoughts.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

# Dense Vectors 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Sparse Vectors
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# Hybrid Search
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever],
    weights=[0.4, 0.6]  # keyword vs semantic
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)


rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system", "Answer the question using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't know'. "
        "Answer in one concise sentence."
    ),
    (
        "human","Answer the question based on the Context:\n{context}\n\nQuestion:\n{question}"
    )
])

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

query = input("Ask a query: ")
response = chain.invoke(query)

print(response)