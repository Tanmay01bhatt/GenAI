from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

sentiment_template = PromptTemplate(
    input_variables=["feedback"],
    template="Determine the sentiment of this feedback and reply in one word as either 'Positive', 'Neutral', or 'Negative':\n\n{feedback}"
)

chain = sentiment_template | llm | StrOutputParser()

user_feedback = "I was extremely disappointed with the customer service. The representative was unhelpful and rude."

feeback_sentiment = chain.invoke({"feedback": user_feedback})

print(feeback_sentiment)