
'''
    Gives the responsibility of joke prompting to an agent instead of imperatively prompting the model.
    Main points:
    - Create a simple agent that generates jokes about a topic
    - Create a graph with the agent (workflow)
    graph.invoke() will return the state of the agent after the execution

    Differences between language models:
    - As previously noted, Cohere can't really work with schemas
'''

import os
from dotenv import load_dotenv
from langchain_cohere import ChatCohere 
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)
from langgraph.graph import END, StateGraph
from typing import List, TypedDict
# Enables typing of the structured LLM model
from langchain_core.runnables.base import RunnableSequence

load_dotenv()

cohere_chat_model = ChatCohere(cohere_api_key=os.getenv("COHERE_API_KEY"))
openai_chat_model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
)


# Select which models you want to use
use_cohere = True
use_openai = False


FUNNY_LLM_PROMPT = ChatPromptTemplate.from_template(
    """
    You task is to make funny jokes about given topic.
    Topic: {topic}                                                      
    """
)

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


# Agents state
class AgentState(TypedDict):
    messages: List[str]
    joke_topic: str
    generated_joke: str
    # The state holds the LLM model to be used
    LLM_model: RunnableSequence

def joker_agent(state: AgentState) -> AgentState:
    # Use created schema to structure the output
    prompt = FUNNY_LLM_PROMPT.format(topic=state["joke_topic"])
    # Invoke the LLM with a prompt and get the structured output
    res = state["LLM_model"].invoke(prompt)
    # Store the result in the state
    state["messages"] += [
        AIMessage(content=f"Generated joke: {res.joke}"),
        AIMessage(content=f"Topic: {res.topic}"),
        AIMessage(content=f"Rating: {res.rating}"),
    ]
    # Store the joke in the state to easily access it later
    state["generated_joke"] = res.joke
    return state

# Create a graph with the state
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("joke", joker_agent)

# Edges
workflow.add_edge("joke", END)

# Set entry point
workflow.set_entry_point("joke")

# Build the graph
graph = workflow.compile()

# Invoke the graph with the state we want to start with
# Just for example we use same "Hello World" as a joke topic and a message
if use_cohere:
    structured_cohere = cohere_chat_model.with_structured_output(FunnySchema)
    res = graph.invoke({"messages": [HumanMessage(content="Hello world")], 
                        "joke_topic": "Hello World", 
                        "LLM_model": structured_cohere})
    # Print the whole state
    print(f"\nCohere's response:\n{res}")
    # Print particular fields from the state
    print(res["messages"])
    print(res["joke_topic"])

    # Print the joke from state
    print(f"\n\n{res["generated_joke"]}")

if use_openai:
    structured_openai = openai_chat_model.with_structured_output(FunnySchema)     
    res = graph.invoke({"messages": [HumanMessage(content="Hello world")], 
                        "joke_topic": "Hello World", 
                        "LLM_model": structured_openai})
    # Print the whole state
    print(f"\nOpenAI's response:\n{res}")
    # Print particular fields from the state
    print(res["messages"])
    print(res["joke_topic"])
    # Print the joke from state
    print(f"\n\n{res["generated_joke"]}")




# The graph will look like this:
#                  +-----------------+
#                  | Start (Entry)   |
#                  +--------+--------+
#                           |
#                           v
#                  +--------+--------+
#                  |  "joke" Node    |
#                  |  (joker_agent)  |
#                  +--------+--------+
#                           |
#                           v
#                  +--------+--------+
#                  |       END        |
#                  +------------------+
