# Conditional Routing for give feedback on user sentiment
# RunnableLambda :converts the custom Python function route into a runnable component that can be used in LangChain chains
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

parse_template = PromptTemplate(
    input_variables=["raw_feedback"],
    template="Parse and clean the following customer feedback for key information:\n\n{raw_feedback}"
)

summary_template = PromptTemplate(
    input_variables=["parsed_feedback"],
    template="Summarize this customer feedback in one concise sentence:\n\n{parsed_feedback}"
)

sentiment_template = PromptTemplate(
    input_variables=["feedback"],
    template="Determine the sentiment of this feedback and reply in one word as either 'Positive', 'Neutral', or 'Negative':\n\n{feedback}"
)

# feedback on user sentiment
thankyou_template = PromptTemplate(
    input_variables=["feedback"],
    template="Given the feedback, draft a thank you message for the user and request them to leave a positive rating on our webpage:\n\n{feedback}"
)

details_template = PromptTemplate(
    input_variables=["feedback"],
    template="Given the feedback, draft a message for the user and request them provide more details about their concern:\n\n{feedback}"
)

apology_template = PromptTemplate(
    input_variables=["feedback"],
    template="Given the feedback, draft an apology message for the user and mention that their concern has been forwarded to the relevant department:\n\n{feedback}"
)

thankyou_chain = thankyou_template | llm | StrOutputParser()
details_chain = details_template | llm | StrOutputParser()
apology_chain = apology_template | llm | StrOutputParser()

def route(info):
    if "postive" in info['sentiment'].lower():
        return thankyou_chain
    elif "negative" in info['sentiment'].lower():
        return apology_chain
    else:
        return details_chain

format_parsed_output = RunnableLambda(lambda output: {"parsed_feedback": output})

summary_chain = parse_template | llm | format_parsed_output | summary_template | llm | StrOutputParser()
sentiment_chain = sentiment_template| llm | StrOutputParser()

user_feedback = "The delivery was late, and the product was damaged when it arrived. However, the customer support team was very helpful in resolving the issue quickly."

# outputs of these chain is a dictionary.
summary = summary_chain.invoke({'raw_feedback' : user_feedback})
sentiment = sentiment_chain.invoke({'feedback': summary})
''' 
a chain output mmay contain multiple keys and values:

sentiment chain op :{
  "feedback": "The product is easy to use and very helpful.",
  "sentiment": "Positive"
}
''' 
print("The summary of the user's message is:", summary)
print("The sentiment was classifed as:", sentiment)

full_chain = {"feedback": lambda x: x['feedback'], 'sentiment' : lambda x : x['sentiment']} | RunnableLambda(route)
#"sentiment": lambda x: x['sentiment'] : {  'sentiment': "Positive"}

# a chain (or runnable) typically takes a dictionary as input and produces a dictionary as output
print(full_chain.invoke({'feedback': summary, 'sentiment': sentiment}))