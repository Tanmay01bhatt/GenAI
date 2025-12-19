#Generating Context-Driven Responses

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o") # Initialize the language model with the specified model

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} # Pass the context and question
    | custom_rag_prompt # Format the prompt using the custom RAG prompt template
    | llm # Use the language model to generate a response
    | StrOutputParser() # Parse the output to a string
)

# without using chain:
'''
passthrough_output = RunnablePassthrough().invoke("Question text")
retriever_output = retriever.invoke({"context": retriever_context, "question": passthrough_output})

custom_prompt_output = custom_rag_prompt.invoke(retriever_output)

llm_output = llm.invoke(custom_prompt_output)

final_output = StrOutputParser().invoke(llm_output)
'''


# Invoke the RAG chain with a question
response = rag_chain.invoke("What is the future of AI?")
print(response) # Print the response


# rag eval
#There are comprehensive evaluation frameworks like Ragas and DeepEval that can aid in this process