#first API call
from openai import OpenAI

API_KEY = "{{API_KEY}}"

client = OpenAI(
  api_key=API_KEY,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant that translates English to Swahili."},
    {"role": "user", "content": "Translate the following English text to Swahili: 'Hello, how are you?'"}
]
model = "gpt-3.5-turbo"

completion = client.chat.completions.create(
  model=model,
  messages=messages
)

print(completion.choices[0].message.content)