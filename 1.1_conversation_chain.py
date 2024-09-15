'''
    A continuous conversation using user input.
    Note: May be token expensive if the conversation grows long.

    Differences between language models:
    - Minimal. Openai tends to be more talkative, Cohere goes straight to the point.
'''

import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere

load_dotenv()


# Select which models you want to use. If both are selected, one is started after previous ends
use_cohere = True
use_openai = False

# Keyword which will break out of the conversation loop
break_word = "Quit"


if use_cohere:
    api_key = os.getenv("COHERE_API_KEY")
    cohere_chat_model = ChatCohere(cohere_api_key=api_key)
    # For storing the existing converstaion
    conversation_history = []
    print("\n/--------------- Starting conversation loop with Cohere:")
    while(True):    
        user_input = input("\n> ")
        if user_input.lower() == break_word.lower():
            print("Exiting conversation loop")
            break

        # Add the newest user prompt to the conversation history. Mark it as Human input with 'HumanMessage'
        conversation_history.append(HumanMessage(content=user_input))
        # Use the entire history this far
        response = cohere_chat_model.invoke(conversation_history)
        # Print response of the model
        print(response.content)
        # Add the latest AI reponse to the history. Mark it as AI with 'AIMessage'
        conversation_history.append(AIMessage(content=response.content))


if use_openai:
    api_key = os.getenv("OPENAI_API_KEY")
    openai_chat_model = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
    )
    # For storing the existing converstaion
    conversation_history = []
    print("\n/--------------- Starting conversation loop with OpenAI:")
    while(True):    
        user_input = input("\n> ")
        if user_input.lower() == break_word.lower():
            print("Exiting conversation loop")
            break

        # Add the newest user prompt to the conversation history. Mark it as Human input with 'HumanMessage'
        conversation_history.append(HumanMessage(content=user_input))
        # Use the entire history this far
        response = openai_chat_model.invoke(conversation_history)
        # Print response of the model
        print(response.content)
        # Add the latest AI reponse to the history. Mark it as AI with 'AIMessage'
        conversation_history.append(AIMessage(content=response.content))