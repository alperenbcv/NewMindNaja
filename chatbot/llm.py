import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o"
)

embedding_model = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
