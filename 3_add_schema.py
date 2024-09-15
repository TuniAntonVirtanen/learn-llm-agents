'''
    Adds a schema to dissect the output of the language model.
    Separates parts of the response into predetermined pieces for later use.

    Differences between language models:
    - Cohere fails majority of tries, but may occasionally succeed partially. Full success is rare.
    - Cohere seems to attempt categorizing the prompt rather than the generated result.
    - OpenAI succeeds most of the time completely.
    - OpenAI seems to prioritize the Schema over prompt (Tested with coherent schema and nonsensical prompt)
'''


import os
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional     # Enables the LLM to return only partially dissected response

load_dotenv()


# Select which models you want to use
use_cohere = True
use_openai = False

FUNNY_LLM_PROMPT = ChatPromptTemplate.from_template(
    """
    You are the funniest person in the world, a comedian, a joker. You make up jokes about every topic.
    Topic: {topic}                                                      
    """
)
# Used for testing purposes. Cohere seems to prioritize prompt while OpenAI seems to prioritize schema.
NONSENSE_LLM_PROMPT = ChatPromptTemplate.from_template(
    """
    Gargoyles. The oozing of, and then there it is.
    Topic: {topic}                                                      
    """
)

joke_topic = "Cat"
# You can change FUNNY_LLM_PROMPT to NONSENSE_LLM_PROMPT here if you want to test how it affects the LLM's behaviour
prompt = FUNNY_LLM_PROMPT.format(topic=joke_topic)

# Create a Pydantic model for the prompt
# Reason of this is to structure the output of the LLM
# Note: Apparently docstring affects outcome of schema "TIP beyond just structure..."
# https://python.langchain.com/v0.2/docs/how_to/structured_output/
# Optional fields may leave the field as None if LLM fails populating it. 
# Failed fields can still be accessed, but their value is None
class FunnySchema(BaseModel):
    """Joke to be told to the user"""   
    # ^-- This is a docstring (First multiline comment as very first thing of class / function)
    topic: Optional[str] = Field(
        description="The topic of the joke",
    )
    joke: Optional[str] = Field(
        description="The joke",
    )
    rating: Optional[int] = Field(
        description="The rating of the joke, from 1 to 10 (bigger is funnier)",
    )
    rating_reason: Optional[str] = Field(
        description="Why the joke is rated this way",
    )


if use_cohere:
    api_key = os.getenv("COHERE_API_KEY")
    cohere_chat_model = ChatCohere(cohere_api_key=api_key)

    # Use created schema to structure the output
    structured_llm = cohere_chat_model.with_structured_output(FunnySchema)

    # Invoke the LLM with a prompt
    response = structured_llm.invoke(prompt)

    # Print the result
    print("\n/------- Cohere's response:\n")
    if response == None:
        print("Response failed")
    else:
        print(f"{response}\n")
        print(response.topic)
        print(response.joke)
        print(response.rating)
        print(response.rating_reason)


if use_openai:
    api_key = os.getenv("OPENAI_API_KEY")
    openai_chat_model = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
    )

    # Use created schema to structure the output
    structured_llm = openai_chat_model.with_structured_output(FunnySchema)

    # Invoke the LLM with a prompt
    response = structured_llm.invoke(prompt)

    # Print the result
    print("\n/------- OpenAI's response:\n")
    if response == None:
        print("Response failed")
    else:
        print(f"{response}\n")
        print(response.topic)
        print(response.joke)
        print(response.rating)
        print(response.rating_reason)

