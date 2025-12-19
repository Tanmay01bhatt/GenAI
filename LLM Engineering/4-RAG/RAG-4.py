#Creating Augmented Queries

from langchain_core.prompts import PromptTemplate

# Define a template for generating answers using provided context
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say 'thanks for asking!' at the end of the answer.

{context}
Question: {question}

Helpful Answer:"""

# Create a custom prompt template using the defined template
custom_rag_prompt = PromptTemplate.from_template(template)
print(custom_rag_prompt) # Print the custom prompt template

# Assume retriever is already defined and configured
question = "What is the future of AI?"
context = retriever.invoke(question)  # Retrieve the context based on the question
print(context)

# Manually format the prompt template to see the augmented query
augmented_query = custom_rag_prompt.format(context=context, question=question)
print("Augmented Query:")
print(augmented_query)


#MMR (Maximal Marginal Relevance) is used to balance the relevance and diversity of the search results, ensuring a varied yet relevant set of documents.