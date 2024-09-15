'''
    This example adds a joke generation agent that combines user input and external data fetched via API using RAG
    Run this script: chainlit run 8_chainlit_api_agent.py
    Main points:
    - Create an agent that fetches a person's name from an external API
    - Combine the user-provided topic and fetched person name to generate a personalized joke
    - Use Chainlit to manage the chatbot interface and interaction
    - Use a state graph to manage the agent's workflow, starting with an API call and followed by the joke generation
    - The final joke is personalized and structured with a rating before being sent to the user via Chainlit
'''


import os
import chainlit as cl
import requests
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from typing import List, TypedDict
from dotenv import load_dotenv


load_dotenv()



api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
)


FUNNY_LLM_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a master comedian, renowned for crafting clever and witty jokes tailored to any audience. 
    Your task is to create a unique and hilarious joke that perfectly combines the topic: {topic} and the person named {name}. 
    Make sure the joke is personal, relevant, and hits just the right comedic note.
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
        description="The rating of the joke, from 1 to 10 (bigger is funnuer)",
    )
    rating_reason: str = Field(
        description="Why the joke is rated this way",
    )

class AgentState(TypedDict):
    messages: List[str]
    joke_topic: str
    generated_joke: str
    person_name: str


async def api_agent(state: AgentState) -> AgentState:
    url = "https://jsonplaceholder.typicode.com/users/1"
    response = requests.get(url)
    await cl.Message(content=f"API response: {response.json()}").send()
    state["messages"] += AIMessage(content=f"API response: {response.json()}")
    state["person_name"] = response.json()["name"]
    return state


def joker_agent(state: AgentState) -> AgentState:
    # Use created schema to structure the output
    structured_llm = llm.with_structured_output(FunnySchema)
    prompt = FUNNY_LLM_PROMPT.format(
        topic=state["joke_topic"], name=state["person_name"]
    )
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
workflow.add_node("api", api_agent)
workflow.add_node("joke", joker_agent)

# Edges
workflow.add_edge("api", "joke")
workflow.add_edge("joke", END)

# Set entry point
workflow.set_entry_point("api")

# Build the graph
graph = workflow.compile()


# CHAINLIT - first message when chat starts
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="Hello! I am a funny chatbot. I can make jokes about any topic. What topic would you like me to make a joke about?"
    ).send()

# Print the whole state
# chainlit - send the joke to the user
@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    print(message.content)
    # first invoke should have something to add to the state

    res = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=message.content)],
            "joke_topic": message.content,
        }
    )

    # Send the joke to the user (which is stored in the state)
    await cl.Message(content=res["generated_joke"]).send()


# GRAPH:
#      +---------+
#      |  api    | <---- Entry Point
#      +---------+
#           |
#           v
#      +---------+
#      |  joke   |
#      +---------+
#           |
#           v
#          END
