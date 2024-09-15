'''
    Adding a prompt for the LLM.
    The prompt will dictate the general behaviour of the LLM.

    Differences between language models:
    - OpenAis more natural talkativeness seems to give more varied results.
    - On one pass the differences are minuscule
'''

import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI

load_dotenv()

# Select which models you want to use
use_cohere = True
use_openai = False


# Create a prompt template, topic is a variable.
FUNNY_LLM_PROMPT = ChatPromptTemplate.from_template(
    """
    You are the funniest person in the world, a comedian, a joker. You make up jokes about every topic.
    Topic: {topic}                                                      
    """
)

# Select the topic for the template
topic = "Cat"

if use_cohere:
    api_key = os.getenv("COHERE_API_KEY")
    cohere_chat_model = ChatCohere(cohere_api_key=api_key)

    # Invoke the LLM with a prompt
    response = cohere_chat_model.invoke(FUNNY_LLM_PROMPT.format(topic=topic))
    # Print the result
    print(f"\nCohere's response:\n{response.content}\n")


if use_openai:
    api_key = os.getenv("OPENAI_API_KEY")
    openai_chat_model = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
    )

    # Invoke the LLM with a prompt
    response = openai_chat_model.invoke(FUNNY_LLM_PROMPT.format(topic=topic))
    # Print the result
    print(f"\nOpenAI's response:\n{response.content}\n")