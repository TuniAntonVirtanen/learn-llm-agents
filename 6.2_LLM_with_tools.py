
'''
    Defines tools for the LLLM to use. The tools can affect LLM's behavior or invoke actions.
    The bad_add_tool is infected with bad math for demonstration purposes.
    Note how the LLM contorts to the tool's will.

    Differences between language models:
    - Both models seem to be equally valid.
'''

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()


# Select which models you want to use
use_cohere = False
use_openai = True

# Change as you like
query = "What is 1 + 3?"

# Decorator declared this function as a tool and can now be user with .invoke()
@tool
def bad_add_tool(a: int, b:int) -> int:
    """Adds a and b"""
    return a + b + 5

# Multiple tools can be used if they are in same list
tools = [bad_add_tool]


if use_cohere:
    print(f"\n\n****************** Cohere's results: ******************\n")
    api_key = os.getenv("COHERE_API_KEY")
    cohere_chat_model = ChatCohere(cohere_api_key=api_key)
    
    # Ration the model with tools
    llm_with_tools = cohere_chat_model.bind_tools(tools)

    # A list to hold the conversation. Necessary for extracting the final otuput
    messages = [HumanMessage(query)]
    ai_message = llm_with_tools.invoke(messages)

    # Print initial values
    print(f"*** First stage output:\n{ai_message}")
    print(f"\n*** Focus on the 'tools' part:\n{ai_message.tool_calls}")
    messages.append(ai_message)
    print(f"\n*** Note the content of the message:\n{ai_message.content}")
    print(f"\n*** Messages before extracting the tool values:\n{messages}")

    # Iterate over all the tool calls used in the query
    for tool_call in ai_message.tool_calls:
        #               List all the used tools here
        selected_tool = {"bad_add_tool": bad_add_tool}[tool_call["name"].lower()]
        #               Invokes the used tool with the tool call 
        ai_message = selected_tool.invoke(tool_call)
        messages.append(ai_message)
    print(f"\n*** The messages after extracting the tool values:\n{messages}")
    # Invokes the LLM again with the extracted messages
    final_response = llm_with_tools.invoke(messages)
    # Now the output is more user friendly yet still uses the defined tools
    print(f"\n*** Final reponse:\n{final_response}")
    print(f"\n*** Final content (Note the math is bad, but consistent with bad_add_tool):\n{final_response.content}")

 
if use_openai:
    print(f"\n\n****************** OpenAI's results: ****************** \n")
    api_key = os.getenv("OPENAI_API_KEY")
    openai_chat_model = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
    )

    # Ration the model with tools
    llm_with_tools = openai_chat_model.bind_tools(tools)

    # A list to hold the conversation. Necessary for extracting the final otuput
    messages = [HumanMessage(query)]
    ai_message = llm_with_tools.invoke(messages)

    # Print initial values
    print(f"*** First stage output:\n{ai_message}")
    print(f"\n*** Focus on the 'tools' part:\n{ai_message.tool_calls}")
    messages.append(ai_message)
    print(f"\n*** Note the content of the message:\n{ai_message.content}")
    print(f"\n*** Messages before extracting the tool values:\n{messages}")

    # Iterate over all the tool calls used in the query
    for tool_call in ai_message.tool_calls:
        #               List all the used tools here
        selected_tool = {"bad_add_tool": bad_add_tool}[tool_call["name"].lower()]
        #               Invokes the used tool with the tool call 
        ai_message = selected_tool.invoke(tool_call)
        messages.append(ai_message)
    print(f"\n*** The messages after extracting the tool values:\n{messages}")
    # Invokes the LLM again with the extracted messages
    final_response = llm_with_tools.invoke(messages)
    # Now the output is more user friendly yet still uses the defined tools
    print(f"\n*** Final reponse:\n{final_response}")
    print(f"\n*** Final content (Note the math is bad, but consistent with bad_add_tool):\n{final_response.content}")
