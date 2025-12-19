import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

st.set_page_config(
    page_title="PDF RAG Q/A Bot",
)
st.title("RAG-based PDF Question Answering")

def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Creating vector database..."):

            st.session_state.embeddings =HuggingFaceEmbeddings(
                 model_name="sentence-transformers/all-MiniLM-L6-v2")


            loader = PyPDFLoader('attention.pdf')
            docs = loader.load()


            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs[:20])


            st.session_state.vectors = FAISS.from_documents(
                chunks,
                st.session_state.embeddings
            )
            st.success("Vector Store DB is ready")

query= st.text_input("Ask your question from the document")

if st.button("Create Document Embeddings"):
    vector_embedding()

if query and "vectors" in st.session_state:

    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

    prompt = PromptTemplate.from_template("""
You are an AI assistant.
Answer ONLY using the provided context.

Context:
{context}

Question:
{question}
""")

    retriever = st.session_state.vectors.as_retriever()

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(query)
    st.subheader("Answer")
    st.write(answer)

    with st.expander("🔍 Document Similarity Search"): #optional = debugging
      docs = retriever.invoke(query)
      for i, doc in enumerate(docs):
        st.markdown(f"**Chunk {i+1}**")
        st.write(doc.page_content)
        st.write("---")