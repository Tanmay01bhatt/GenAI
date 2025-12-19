# Langchiain RAG
#indexing
import os
os.environ['OPENAI_API_KEY'] = '{{OpenAI_API}}' 

from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings() # Initialize the embeddings model
# Generate embeddings for a list of documents
embeddings = embeddings_model.embed_documents(
    [
    "This is the Fundamentals of RAG course.",
    "Educative is an AI-powered online learning platform.",
    "There are several Generative AI courses available on Educative.",
    "I am writing this using my keyboard.",
    "JavaScript is a good programming language"
    ]
)

print(len(embeddings)) # Print the number of embeddings generated (should be equal to the number of documents)
print(len((embeddings[0]))) # Print the length of the first embedding vector

# retrieval
# List of example documents to be used in the database
documents = [
    "Python is a high-level programming language known for its readability and versatile libraries.",
    "Java is a popular programming language used for building enterprise-scale applications.",
    "JavaScript is essential for web development, enabling interactive web pages.",
    "Machine learning is a subset of artificial intelligence that involves training algorithms to make predictions.",
    "Deep learning, a subset of machine learning, utilizes neural networks to model complex patterns in data.",
    "The Eiffel Tower is a famous landmark in Paris, known for its architectural significance.",
    "The Louvre Museum in Paris is home to thousands of works of art, including the Mona Lisa.",
    "Artificial intelligence includes machine learning techniques that enable computers to learn from data.",
]

# Create a Chroma database from the documents using OpenAI embeddings
db = Chroma.from_texts(documents, OpenAIEmbeddings())
print(db)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1}
)

result = retriever.invoke("Where can I see Mona Lisa?")
print(result)