import requests
from dotenv import load_dotenv
import os
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

mcp = FastMCP("WeatherAssistant")

API_KEY = os.getenv("OPENWEATHER_API_KEY")

@mcp.tool()
def get_weather(location: str) -> dict:
    if not API_KEY:
        return {"error": "API key not set"}

    try:
        res = requests.get(
            "http://api.openweathermap.org/data/2.5/weather",
            params={"q": location, "appid": API_KEY, "units": "metric"},
            timeout=5
        )
        res.raise_for_status()
        data = res.json()

        return {
            "location": data.get("name"),
            "weather": data["weather"][0]["description"],
            "temp": f"{data['main']['temp']}°C",
        }

    except requests.exceptions.HTTPError:
        return {"error": f"HTTP {res.status_code}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")