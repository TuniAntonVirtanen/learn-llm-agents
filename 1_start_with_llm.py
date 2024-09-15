'''
    A single response from hard coded input

    Differences between language models:
    - Nonexistent
'''
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI

load_dotenv()


# Select which models you want to use
use_cohere = True
use_openai = False

# Select a prompt for the models
prompt = "Hello World"

if use_cohere:
    # Initialize the language model
    api_key = os.getenv("COHERE_API_KEY")
    cohere_chat_model = ChatCohere(cohere_api_key=api_key)

    # invoke the LLM
    response = cohere_chat_model.invoke(prompt)

    # Print the result
    print(f"\nCohere response:\n{response.content}")


if use_openai:
    # Initialize the language model
    api_key = os.getenv("OPENAI_API_KEY")
    openai_chat_model = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
    )

    # invoke the LLM
    response = openai_chat_model.invoke(prompt)

    # Print the result
    print(f"\nOpenai response:\n{response.content}")