#sentiment analyzer
from openai import OpenAI

API_KEY = "{{API_KEY}}"

client = OpenAI(
  api_key=API_KEY,
)


TEXT_TO_ANALYZE = "Replace this with the text you want to analyze."

def analyze_sentiment(text):
    """Analyze sentiment of the given text."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Analyze the sentiment of the following text: \"{text}\". Is it positive, negative, or neutral? Answer in one word with no punctuation."}
        ],
        max_tokens=50
    )
    sentiment = response.choices[0].message.content.strip().lower()

    # Validate sentiment to ensure it's one of the three categories
    if sentiment not in ["positive", "negative", "neutral"]:
        return "Unable to determine sentiment. Please try again."
    
    return f"The sentiment of the text is: {sentiment}"

result = analyze_sentiment(TEXT_TO_ANALYZE)
print(result)