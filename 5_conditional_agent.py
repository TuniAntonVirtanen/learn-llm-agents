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

# This example add an other agent, which job is to improve the joke if needed (conditional agent)
# Main points:
# - Create a conditional agent that improves the joke if the rating is low
# - Add conditional edge to the graph
# - Use the state of the agent to determine the next steps (END or improve the joke)
# - loop the agents until the joke is funny enough (max 5 iterations, so not to loop forever)


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

IMPROVER_LLM_PROMPT = ChatPromptTemplate.from_template(
    """
    You know everything about jokes. You can improve any joke. You can make any joke funnier.
    You task is to improve the joke about the topic: {topic}
    
    Original joke: {joke} 
    
    Your job is to make better topic so joke can be improved.                                         
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
    
class ImprovedJokeSchema(BaseModel):
    suggestions: List[str] = Field(
        description="Suggestions to improve the joke",
    )
    new_topic: str = Field(
        description="The new improved topic of the joke, which is generated using the original topic and suggestions",
    )


# Agents state
class AgentState(TypedDict):
    messages: List[str]
    joke_topic: str
    generated_joke: str
    joke_rating: int
    iteration: int


def joker_agent(state: AgentState) -> AgentState:
    print(f"\n**Joker Agent**")
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
    state["joke_rating"] = res.rating
    print(f"Joke: {res.joke}")
    print(f"Joke rating: {res.rating}")
    return state

# This agent will improve the joke if the orginal joke is not funny enough (used in the conditional edge)
def joke_improver_agent(state: AgentState) -> AgentState:
    print(f"\n**Joke Improver Agent**")
    # Use created schema to structure the output
    structured_llm = llm.with_structured_output(ImprovedJokeSchema)
    prompt = IMPROVER_LLM_PROMPT.format(topic=state["joke_topic"], joke=state["generated_joke"])
    # Invoke the LLM with a prompt and get the structured output
    res = structured_llm.invoke(prompt)
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
    

# Create a graph with the state
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("joke", joker_agent)
workflow.add_node("joke_improver", joke_improver_agent)

# Edges
# workflow.add_edge("joke", END)
workflow.add_edge("joke_improver", "joke")

def is_done(state):
    # Determ next steps after the first run
    if state["joke_rating"] < 5:
        if state["iteration"] > 5:
            return END
        return "joke_improver"
    else:
        return END

# Add conditional edge
workflow.add_conditional_edges("joke", is_done)

# Set entry point
workflow.set_entry_point("joke")

# Build the graph
graph = workflow.compile()

# Invoke the graph with the state we want to start with
# Just for example we use same "Hello World" as a joke topic and a message
res = graph.invoke({"messages": [HumanMessage(content="Not funny Hello world joke")], "joke_topic": "Not funny Hello World joke","iteration": 0})

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



# Print the whole state
print(f"\n\n{res}")
# Print particular fields from the state
print(res["messages"])
print(res["joke_topic"])

# Print the joke from state
print(f"\n\n{res["generated_joke"]}")


