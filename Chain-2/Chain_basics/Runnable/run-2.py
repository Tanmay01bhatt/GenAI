# Adding Chains/Extending a Chain = using Runnable Lambda
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

# step-1
parse_template = PromptTemplate(
    input_variables=["raw_feedback"],
    template="Parse and clean the following customer feedback for key information:\n\n{raw_feedback}"
)
#step-2
summary_template = PromptTemplate(
    input_variables=["parsed_feedback"],
    template="Summarize this customer feedback in one concise sentence:\n\n{parsed_feedback}"
)
#step-3
sentiment_template = PromptTemplate(
    input_variables=["feedback"],
    template="Determine the sentiment of this feedback and reply in one word as either 'Positive', 'Neutral', or 'Negative':\n\n{feedback}"
)
#intermediate outputs
format_parsed_output = RunnableLambda(lambda output: {"parsed_feedback": output})
format_summary_output = RunnableLambda(lambda output: {"feedback": output})

chain = parse_template | llm | format_parsed_output | summary_template | llm | format_summary_output | sentiment_template| llm | StrOutputParser()

user_feedback = "The customer service was fantastic. The representative was friendly, knowledgeable, and resolved my issue quickly."
feedback_sentiment = chain.invoke({"raw_feedback": user_feedback})

print(feedback_sentiment)