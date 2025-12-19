# tools.py
from langchain_community.tools import DuckDuckGoSearchRun, YouTubeSearchTool #tool that performs searches on YouTube
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,    #Tool that performs real-time web searches
    WikipediaAPIWrapper,           #Allows LangChain to fetch information from Wikipedia
    ArxivAPIWrapper,               #Allows the agent to search arXiv, a repository of scientific papers  
    OpenWeatherMapAPIWrapper,     #Enables the agent to retrieve weather data from OpenWeatherMap
) 

import requests

# -----------------------
#  Multi Web Search
# -----------------------
def multi_web_search(query: str) -> str:
    """Run DuckDuckGo, Wikipedia, and Arxiv searches together."""
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(backend="lite")   #Initialize all API wrappers
        ddg_search = DuckDuckGoSearchRun(api_wrapper=wrapper)
        wiki = WikipediaAPIWrapper()
        arxiv = ArxivAPIWrapper()

        results = []
        results.append("DuckDuckGo: " + ddg_search.run(query))   #Perform the searches and Prefixes each result with a label and Stores them in a list
        results.append("Wikipedia: " + wiki.run(query))
        results.append("Arxiv: " + arxiv.run(query))

        return "\n\n".join(results)      #Joins all results with blank lines between them
    except Exception as e:
        return f"Search error: {e}"      #If any API call fails, returns an error message
    

# -----------------------
#  Video Search
# -----------------------
def youtube_search(query: str) -> str:   #Accepts a search query (string) and returns a string result
    """Search YouTube for videos."""
    try:
        yt = YouTubeSearchTool()     #Initializes the YouTube search tool
        return yt.run(query)            #Performs the youtube search and returns the results
    except Exception as e:
        return f"YouTube error: {e}"

#  Weather Tool
# -----------------------
def weather_search(query: str, api_key: str) -> str:
    """Get weather information using OpenWeather API."""
    try:
        weather = OpenWeatherMapAPIWrapper(openweathermap_api_key=api_key)  #  Initializes the OpenWeatherMap API wrapper with the provided API key
        return weather.run(query)        #sends the query directly to OpenWeatherMap and query is interpreted as a location.
    except Exception as e:
        return f"Weather error: {e}"

# -----------------------
#  News Tool
# -----------------------
def news_search(query: str, api_key: str) -> str:
    url = "https://newsapi.org/v2/everything"   #using the everything endpoint, which supports: full-text search and filtering by date, language, domains, etc.
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": 5,
    }
    resp = requests.get(url, params=params, timeout=10)  # Sends a GET request with a 10-second timeout
    resp.raise_for_status()              # Raises an error for bad responses
    data = resp.json()              # Parses the JSON response
    articles = data.get("articles", [])     # Extracts the list of articles from the response dictionary(get the value under the key "articles")
    out = []
    for art in articles:                    # Format the results (selecting only two feildsfrom the enitre article)
        title = art.get("title")  
        url = art.get("url")
        out.append(f"{title} — {url}")
    return "\n".join(out) or "No news found."   # Joins the formatted articles into a single string or If the list is empty, returns "No news found."
