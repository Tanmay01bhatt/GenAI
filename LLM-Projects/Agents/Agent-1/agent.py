# agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory  #Adds memory to your agent so it can remember earlier messages.
from tools import (
    multi_web_search,
    youtube_search,
    weather_search,
    news_search,
)

# -----------------------
#  API Keys
# -----------------------
GOOGLE_API_KEY =  "google api key"  # free
OPENWEATHER_API_KEY = "open weather api key"  # free
NEWS_API_KEY = "news api key"  # free

# -----------------------
# Setup LLM
# -----------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,   # make it a bit conversational
)


# -----------------------
# Wrap Tools
# -----------------------
tools = [
    Tool(
        name="MultiWebSearch",
        func=multi_web_search,
        description="Search DuckDuckGo, Wikipedia, and Arxiv together for information.",
    ),
    Tool(
        name="YouTubeSearch",
        func=youtube_search,
        description="Search YouTube for videos on a given topic.",
    ),
    Tool(
        name="WeatherSearch",
        func=lambda q: weather_search(q, OPENWEATHER_API_KEY),   #LangChain tools must be single-argument functions, so we use a lambda to pass the API key
        description="Get current weather information for a city/location.",
    ),
    Tool(
        name="NewsSearch",
        func=lambda q: news_search(q, NEWS_API_KEY),
        description="Fetch latest news for a given topic.",
    ),
]

# -----------------------
# Add Memory
# -----------------------
memory = ConversationBufferMemory(  #stores the entire conversation history between the user and the agent.
    memory_key="chat_history", 
    return_messages=True
)

# -----------------------
# Initialize Agent
# -----------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # conversational agent type
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
)

# -----------------------
# Function for UI :  bridge between your UI (frontend) and your LangChain agent.
# -----------------------
def get_agent_response(query: str) -> str:
    """Takes a user query and returns the agent's response."""
    try:
        return agent.run(query)
    except Exception as e:
        return f"Agent error: {e}"

