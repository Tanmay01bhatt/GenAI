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
import streamlit as st

load_dotenv()

def data_ingestion(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    return chunks

def vector_embedding(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

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
    return rag_chain

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#temp dir
import os

def _get_file_path(file_upload):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file_upload.name)
    with open(file_path, "wb") as f:
        f.write(file_upload.getbuffer())

    return file_path    

# app
st.set_page_config(page_title="RAG Chat Bot")
st.title("RAG Chatbot with Chat History")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_ui" not in st.session_state:
    st.session_state.chat_ui = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "user-1"

uploaded_file = st.file_uploader( "Upload a PDF",type=["pdf"],key="pdf_uploader")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        file_path = _get_file_path(uploaded_file)
        chunks = data_ingestion(file_path)
        st.session_state.vectorstore = vector_embedding(chunks)
        st.success("PDF processed and vector store created.")
        rag_chain = rag_chain(st.session_state.vectorstore)

        st.session_state.chat_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history")
        
        

    st.success("PDF processed successfully!")


for role, msg in st.session_state.chat_ui:
    with st.chat_message("user" if role == "You" else "assistant"):
        st.markdown(msg)

query = st.chat_input("Ask a question about the PDF")

with st.spinner("Generating response..."):
    if query:
    # Save user message
        st.session_state.chat_ui.append(("You", query))

        response = st.session_state.chat_rag_chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": st.session_state.session_id}}
        )

        # Save bot message
        st.session_state.chat_ui.append(("Bot", response))

    # 🔑 Force rerun so full history re-renders
        st.rerun()
