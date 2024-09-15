'''
    Adds multiple schemas to dissect the output of the language model.
    LLM Decides which schema to use per prompt and generated response.

    Notes:
    - The order by which schemas are added into CombinedSchema Union seems to matter. First one takes priority

    Differences between language models:
    - OpenAI succeeds acceptably after prioritizing simpler schemas over complex ones.
    - Cohere doesn't seem to support the common parent and union approach of schemas and causes the program to crash.
    - Cohere is not a viable tool here, but OpenAI seems to work.
'''


import os
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional
from typing import Union        # Enables combining schemas

load_dotenv()


# Select which models you want to use
use_cohere = True
use_openai = False

PROMPT = ChatPromptTemplate.from_template(
    """
    You will either tell a joke or answer in normal manner depending on the user's message. Choose appropriate schema accordingly.
    User's message: {message}                                                      
    """
)

# Two different inputs, trying to provoke the use of two different schemas.
chat_message = "What is the capital of Finland"
joke_message = "Tell me a joke about cows"

chat_prompt = PROMPT.format(message=chat_message)
joke_prompt = PROMPT.format(message=joke_message)

# Enable dissecting response if it is a joke...
class FunnySchema(BaseModel):
    """Joke to be told to the user"""    
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
        description="Why the joke is rated thi way",
    )

# ... Offer regular conversational manner schema for the LLM aswell ...
class ConversationalSchema(BaseModel):
    """A regular back and forth conversation with user"""
    response: str = Field(description="A regular response to the user's message")

# ... Adds given schemas to a parent schema. Individual schemas still work independently
class CombinedSchema(BaseModel):
    output: Union[ConversationalSchema, FunnySchema]

if use_cohere:
    api_key = os.getenv("COHERE_API_KEY")
    cohere_chat_model = ChatCohere(cohere_api_key=api_key)

    structured_llm = cohere_chat_model.with_structured_output(FunnySchema)

    # Invoke the LLM with a prompt
    response = structured_llm.invoke(joke_prompt)

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

    structured_llm = openai_chat_model.with_structured_output(CombinedSchema)

    joke_response = structured_llm.invoke(joke_prompt)
    chat_response = structured_llm.invoke(chat_prompt)

    # Print the result. Attribute is checked before it is printed to avoid unnecessary, possible errors
    # Note that the format of multi schema output is different than single schema output. Example at the end of file.
    print("\n/------- OpenAI's response:\n")
    if joke_response == None:
        print("Joke Response failed")
    else:
        output = joke_response.output
        print(f"Joke response:\n{joke_response}\n")
        if hasattr(output, 'topic'):
            print(output.topic)
        if hasattr(output, 'joke'): 
            print(output.joke)
        if hasattr(output, 'rating'):
            print(output.rating)
        if hasattr(output, 'rating_reason'):
            print(output.rating_reason)
    if chat_response == None:
        print("Chat Response failed")
    else:
        output = chat_response.output
        print(f"\nChat response:\n{chat_response}\n")
        if hasattr(output, 'response'):
            print(output.response)


if use_cohere:
    api_key = os.getenv("COHERE_API_KEY")
    cohere_chat_model = ChatCohere(cohere_api_key=api_key)

    try:
        structured_llm = cohere_chat_model.with_structured_output(CombinedSchema)

        joke_response = structured_llm.invoke(joke_prompt)
        chat_response = structured_llm.invoke(chat_prompt)

        # Print the result. Attribute is checked before it is printed to avoid unnecessary, possible errors
        # Note that the format of multi schema output is different than single schema output. Example at the end of file.
        print("\n/------- Cohere's response:\n")
        if joke_response == None:
            print("Joke Response failed")
            
        else:
            output = joke_response.output
            print(f"Joke response:\n{joke_response}\n")
            if hasattr(output, 'topic'):
                print(output.topic)
                
            if hasattr(output, 'joke'): 
                print(output.joke)
            if hasattr(output, 'rating'):
                print(output.rating)
            if hasattr(output, 'rating_reason'):
                print(output.rating_reason)
        
        if chat_response == None:
            print("Chat Response failed")
        else:
            output = chat_response.output
            print(f"\nChat response:\n{chat_response}\n")
            if hasattr(output, 'response'):
                print(output.response)
    except Exception as e:
        print(f"CombinedSchema causes a deep level error when using Cohere:\n{e}")
            


'''
    Example of multi schema output:
    {'output': FunnySchema(topic='cows', joke='Why do cows have hooves instead of feet? Because they lactose!', rating=7, rating_reason="It's a classic pun that plays on the word 'lactose', making it a fun and light-hearted joke.")}

    ie. The used schema is the value of the key 'output'
'''

# A workaround attempt for using Union for Cohere. This doesn't give any better results
class CombinedSchema_hacked(BaseModel):
    """Combined schema to handle either a joke or a conversation"""
    type: str = Field(description="Type of output: 'joke' or 'conversation'")
    joke_data: Optional[FunnySchema] = Field(description="Data for a joke, if the type is 'joke'")
    conversation_data: Optional[ConversationalSchema] = Field(description="Data for a conversation, if the type is 'conversation'")    
