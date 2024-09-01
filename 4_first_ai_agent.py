import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)
from langgraph.graph import END, StateGraph
from typing import List, TypedDict
from dotenv import load_dotenv

# This example is a simple agent that generates jokes about a topic
# Main points:
# - Create a simple agent that generates jokes about a topic
# - Create a graph with the agent (workflow)
# graph.invoke() will return the state of the agent after the execution

# .env file is used to store the api key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# Initialize the language model
# use dotnenv to load OPENAI_API_KEY api key
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


# Create a Pydantic model for the prompt
# Reason of this is to structure the output of the LLM
class FunnySchema(BaseModel):
    topic: str = Field(
        description="The topic of the joke",
    )
    joke: str = Field(
        description="The joke",
    )
    rating: int = Field(
        description="The rating of the joke, from 1 to 10 (bigger is funnuer)",
    )
    rating_reason: str = Field(
        description="Why the joke is rated this way",
    )


# Agents state
class AgentState(TypedDict):
    messages: List[str]
    joke_topic: str
    generated_joke: str


def joker_agent(state: AgentState) -> AgentState:
    # Use created schema to structure the output
    structured_llm = llm.with_structured_output(FunnySchema)
    prompt = FUNNY_LLM_PROMPT.format(topic=state["joke_topic"])
    # Invoke the LLM with a prompt and get the structured output
    res = structured_llm.invoke(prompt)
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
res = graph.invoke({"messages": [HumanMessage(content="Hello world")], "joke_topic": "Hello World"})

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


# Print the whole state
print(res)
# Print particular fields from the state
print(res["messages"])
print(res["joke_topic"])

# Print the joke from state
print(f"\n\n{res["generated_joke"]}")


