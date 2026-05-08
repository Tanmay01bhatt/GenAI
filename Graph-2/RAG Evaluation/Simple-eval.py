from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv

load_dotenv()

# --- Setup ---
docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(
    PyPDFLoader("attention.pdf").load()
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = FAISS.from_documents(docs, embeddings).as_retriever(k=5)
llm = ChatGroq(model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using this context:\n{context}"),
    ("human", "{question}")
])

# --- RAG function ---
def ask(question):
    retrieved = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in retrieved)
    answer = llm.invoke(prompt.format_messages(context=context, question=question)).content
    return answer, retrieved

# --- Eval buffer ---
buffer = []

def run(question):
    answer, retrieved = ask(question)
    buffer.append(SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=[d.page_content for d in retrieved]
    ))
    return answer

def run_eval():
    if not buffer:
        print("Nothing to evaluate."); return
    metrics = [
        Faithfulness(llm=LangchainLLMWrapper(llm)),
        AnswerRelevancy(llm=LangchainLLMWrapper(llm), embeddings=LangchainEmbeddingsWrapper(embeddings))
    ]
    results = evaluate(EvaluationDataset(samples=buffer), metrics=metrics).to_pandas()
    print(results[["user_input", "faithfulness", "answer_relevancy"]].to_string(index=False))
    buffer.clear()

# --- Chat loop ---
print("Type 'eval' to evaluate, 'exit' to quit\n")
while True:
    q = input("You: ")
    if q.lower() == "exit": break
    if q.lower() == "eval": run_eval(); continue
    print("Bot:", run(q), f"\n[{len(buffer)} buffered]\n")