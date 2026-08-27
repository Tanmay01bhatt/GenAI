# generation is context aware but retrival is not(uses the raw question).
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory #Stores messages
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory #fetches chat history from ChatMessageHistory injects it into the chain
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

load_dotenv()

loader = PyPDFLoader('Tree of Thoughts.pdf')
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

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer using the given context."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "Context:\n{context}\nQuestion:\n{question}")
])

rag_chain = (
    {
        "context": RunnableLambda(lambda x: retriever.invoke(x["question"])),
        "question": RunnableLambda(lambda x: x["question"]),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Chat History/Memory
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chat_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

session_id = "user-1"

print("Conversational RAG Bot (type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    response = chat_rag_chain.invoke(
    {"question": query},
    config={"configurable": {"session_id": session_id}}
  )
    print("Bot:", response)
