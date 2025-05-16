from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemma-3-12b-it")

while True:
    query = input("Enter your query: ")
    if query.lower() == "exit":
        break
    else:
        result = llm.invoke(query)
        print(result.content)
