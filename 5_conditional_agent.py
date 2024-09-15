'''
    This example add an other agent, which job is to improve the joke if needed (conditional agent)
    Main points:
    - Create a conditional agent that improves the joke if the rating is low
    - Add conditional edge to the graph
    - Use the state of the agent to determine the next steps (END or improve the joke)
    - loop the agents until the joke is funny enough (max 5 iterations, so not to loop forever)

    Differences between language models:
    - Surprisingly Cohere is able to run this graph sometimes. The result isn't great, but works.
    - Cohere's version of joke_improver isn't anywhere as good as OpenAI's.
'''

import os
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)
from langgraph.graph import END, StateGraph
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_core.runnables.base import RunnableSequence

load_dotenv()

# Select which models you want to use
use_cohere = True
use_openai = False

# Joke topic to be used
joke_topic = "Not funny Hello World joke"


# Create a prompt template, topic is a variable
FUNNY_LLM_PROMPT = ChatPromptTemplate.from_template(
    """
    You are the funniest person in the world, a comedian, a joker. You make up jokes about every topic.
    Topic: {topic}                                                      
    """
)

IMPROVER_LLM_PROMPT = ChatPromptTemplate.from_template(
    """
    You know everything about jokes. You can improve any joke. You can make any joke funnier.
    You task is to improve the joke about the topic: {topic}
    
    Original joke: {joke} 
    
    Your job is to make better topic so joke can be improved.                                         
    """
)


class FunnySchema(BaseModel):
    topic: str = Field(
        description="The topic of the joke",
    )
    joke: str = Field(
        description="The joke",
    )
    rating: int = Field(
        description="The rating of the joke, from 1 to 10 (bigger is funnier)",
    )
    rating_reason: str = Field(
        description="Why the joke is rated this way",
    )
    
class ImprovedJokeSchema(BaseModel):
    suggestions: List[str] = Field(
        description="Suggestions to improve the joke",
    )
    new_topic: str = Field(
        description="The new improved topic of the joke, which is generated using the original topic and suggestions",
    )

class AgentState(TypedDict):
    messages: List[str]
    joke_topic: str
    generated_joke: str
    joke_rating: int
    iteration: int
    LLM_model: RunnableSequence

def joker_agent(state: AgentState) -> AgentState:
    print(f"\n**Joker Agent**")
    prompt = FUNNY_LLM_PROMPT.format(topic=state["joke_topic"])
    structured_LLM = state["LLM_model"].with_structured_output(FunnySchema)
    res = structured_LLM.invoke(prompt)

    try:
        # Store the result in the state
        state["messages"] += [
            AIMessage(content=f"Generated joke: {res.joke}"),
            AIMessage(content=f"Topic: {res.topic}"),
            AIMessage(content=f"Rating: {res.rating}"),
        ]
        # Store the joke in the state to easily access it later
        state["generated_joke"] = res.joke
        state["joke_rating"] = res.rating
        print(f"Joke: {res.joke}")
        print(f"Joke rating: {res.rating}")
        return state
    except Exception as e:
        print(f"The LLM model failed with response:\n{e}\nExiting program")
        exit()


# This agent will improve the joke if the orginal joke is not funny enough (used in the conditional edge)
def joke_improver_agent(state: AgentState) -> AgentState:
    print(f"\n**Joke Improver Agent**")
    prompt = IMPROVER_LLM_PROMPT.format(topic=state["joke_topic"], joke=state["generated_joke"])
    # Invoke the LLM with a prompt and get the structured output
    structured_LLM = state["LLM_model"].with_structured_output(ImprovedJokeSchema)
    res = structured_LLM.invoke(prompt)

    try:
        # Store the result in the state
        state["messages"] += [
            AIMessage(content=f"Joke suggestions for new topic: {res.suggestions}"),
            AIMessage(content=f"New topic for the joke: {res.new_topic}"),
        ]
        # Overwrite the joke topic with the new improved topic, so the joker agent can generate a new joke using it
        state["joke_topic"] = res.new_topic
        state["iteration"] += 1
        print(f"new topic: {res.new_topic}")
        return state
    except Exception as e:
        print(f"The LLM model failed with response:\n{e}\nExiting program")
        exit()

# Create a graph with the state
workflow = StateGraph(AgentState)

def is_done(state):
    # Determ next steps after the first run
    if state["joke_rating"] < 6:
        if state["iteration"] > 3:
            return END
        return "joke_improver"
    else:
        return END

# Nodes
workflow.add_node("joke", joker_agent)
workflow.add_node("joke_improver", joke_improver_agent)

# Edges
workflow.add_edge("joke_improver", "joke")

# Add conditional edge
workflow.add_conditional_edges("joke", is_done)

# Set entry point
workflow.set_entry_point("joke")

# Build the graph
graph = workflow.compile()


if use_cohere:
    print("Running agent with Cohere:\n")    
    cohere_chat_model = ChatCohere(cohere_api_key=os.getenv("COHERE_API_KEY"))
    res = graph.invoke({"messages": [HumanMessage(content="Not funny Hello world joke")], 
                        "joke_topic": joke_topic,
                        "iteration": 0,
                        "LLM_model": cohere_chat_model})
    print(f"\n\n{res}")
    print(res["messages"])
    print(res["joke_topic"])
    print(f"\n\n{res["generated_joke"]}")

if use_openai:
    print("Running agent with OpenAI:\n")    
    openai_chat_model = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    ) 
    res = graph.invoke({"messages": [HumanMessage(content="Not funny Hello world joke")], 
                        "joke_topic": joke_topic,
                        "iteration": 0,
                        "LLM_model": openai_chat_model})
    print(f"\n\n{res}")
    print(res["messages"])
    print(res["joke_topic"])
    print(f"\n\n{res["generated_joke"]}")    

#GRAPH WILL LOOK LIKE THIS
#                  +------------------+
#                  | Start (Entry)    |
#                  +---------+--------+
#                            |
#                            v
#                  +---------+--------+
#                  | "joke" Node      |
#                  | (joker_agent)    |<-------
#                  +---------+--------+       |
#                            |                |
#                            v                |
#        +--------------------+----------------------+
#        |                                    |       |
#   (Rating >= 5)                             |  (Rating < 5)
#        |                                    |       |
#        v                                    |       v
#+-------+-------+                          +--------+---------+
#|      END      |                          | "joke_improver"  |
#+---------------+                          |   Node           |
#                                           | (joke_improver_   |
#                                           |     agent)        |
#                                           +--------+----------+






