from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ('system','you are a professional translator'),
    ('user','Detect the language of the following text and translate it to {target_lang}:\n'
     '\n{format_instructions}'
     '\nText : {text}')
])

class TranslationParser(BaseModel):
    detected_language: str = Field(description="Detected source language")
    translated_text: str = Field(description="Translated text")

parser = PydanticOutputParser(pydantic_object=TranslationParser)

chain = prompt | llm | parser

text = input("Enter text to translate: ")
target_lang = input("Enter target language: ")
result = chain.invoke({
    'text' : text,
    'target_lang' : target_lang,
    'format_instructions' : parser.get_format_instructions()
})

print(result)