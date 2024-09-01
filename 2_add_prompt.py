import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# This example adds a prompt to the LLM

# .env file is used to store the api key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the language model
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
)

# Create a prompt template, topic is a variable
FUNNY_LLM_PROMPT = ChatPromptTemplate.from_template(
    """
    You are the funniest person in the world, a comedian, a joker. You make up jokes about every topic.
    Topic: {topic}                                                      
    """
)

# Invoke the LLM with a prompt
res = llm.invoke(FUNNY_LLM_PROMPT.format(topic="Hello World"))

# Print the result
print(res.content)
