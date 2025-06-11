from langchain_openai import ChatOpenAI
from os import getenv
from dotenv import load_dotenv
from pydantic import SecretStr
import os

load_dotenv()

api = os.getenv("OPENROUTER_API_KEY")
llm = ChatOpenAI(
    api_key=SecretStr(api) if api is not None else None,
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-r1-zero:free",
)

prompt = "Hi there! Vedant this side."

print(llm.invoke(prompt).content)
