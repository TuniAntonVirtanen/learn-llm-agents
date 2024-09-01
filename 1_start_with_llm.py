import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

#This example is basic code to start with LLM

# .env file is used to store the api key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the language model
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
)

# invoke the LLM
res = llm.invoke("Hello World")

# Print the result
print(res.content)
