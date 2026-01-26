import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from dotenv import load_dotenv
import uuid
import os 
load_dotenv()

os.makedirs("pharma_db", exist_ok=True)

file_path = 'mental_health_Document.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()

splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-mpnet-base-v2",
            chunk_size=1000,
            chunk_overlap=100
        )

chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
db_path = "./chroma_db" 
if not os.path.exists(db_path):

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
else:
    vectorstore = Chroma(
        embedding=embeddings,
        persist_directory=db_path
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})
llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2
    )

q_prompt = ChatPromptTemplate.from_messages([
    (
        "system","Given the chat history and the user's latest question, "
        "rewrite the question so it can be understood on its own. "
        "Do not answer the question."
    ),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{query}")
    ])
q_chain = q_prompt | llm | StrOutputParser()

r_prompt = ChatPromptTemplate.from_messages([
        ("system","You are helpful assistant who is great in analysing mental health documents"
         "The responses should be concinse "
         "Only answer based on the information from the documents provided"
        "Do not make up answer.If the context is insufficient say 'Insufficient information'"),
        
        ("user","Answer the following questions based on the the context : {context}"
        "\n\nQuestion : {query}")
        ])
r_chain = q_chain | retriever
rag_chain = (
        {
            "context": r_chain | format_docs,
            "query": RunnablePassthrough(),
            
        }
        | r_prompt
        | llm
        | StrOutputParser()
    )


store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chat_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="chat_history",
)

session_id = str(uuid.uuid4())

# inference
print("\nConversational RAG Bot (type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    response = chat_rag_chain.invoke(
        {"query": query},
        config={"configurable": {"session_id": session_id}}
    )

    print("Bot:", response)
