from openai import OpenAI

API_KEY = "{{API_KEY}}"

client = OpenAI(
  api_key=API_KEY,
)

messages = [
    {
        "role": "system",
        "content": "You are a programming expert. Answer questions about programming languages"
    },
]

user_qestions = [
    "What's a popular choice for a first programming language?",
    "What are some advantages of learning it as my first language?",
    "Can you show me a simple 'Hello World' program written in that language?",
]

for user_question in user_qestions:
    print("User: " + user_question)

    user_message = {"role": "user", "content": user_question}
    messages.append(user_message)

    response = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = messages,
    )
    assistant_message = {"role": "assistant", "content": response.choices[0].message.content}
    messages.append(assistant_message)

    print("Assistant: ", response.choices[0].message.content, "\n")