'''
    This example adds a database agent to the graph
    Main points:
    - Create a database agent that insert a new joke (rating over 5) into the database
    - Database is initialized and some jokes are inserted into it on the start (if not already)
    - Use the database tables and their descriptions in the prompt, so the agent can generate a query to insert the joke
    - If inserted joke is a duplicate, the database will raise an error (and app will end without inserting the joke)
'''


import os
import sqlite3 # for database
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
#DATABASE THINGS
from database import init_db, sql

load_dotenv()

# Select which models you want to use
use_cohere = False
use_openai = True


# Initialize the database, create the table and insert some jokes to it
init_db.initialize_database("database/jokes.db")

tables = sql.list_tables()
description_of_tables = sql.describe_table([tables])

print(f"\nTables in the database: {tables}")
print(f"\nDescription of tables: {description_of_tables}")


FUNNY_LLM_PROMPT = ChatPromptTemplate.from_template(
    """
    You are the funniest person in the world, a comedian, a joker. You make up jokes about every topic. Try to make unique jokes when asked.
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

DATABASE_QUERY_LLM_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a seasoned database expert specializing in crafting optimized SQL-lite queries. Your task is to generate insert query, to insert new joke to database.

    Topic: {topic}
    
    Joke: {joke}
    
    Rating: {rating}

    Available Database Tables: 
    {tables}

    Detailed Table Descriptions and Relationships:
    {table_descriptions}

    Consider the structure and relationships between the tables to ensure the query efficiently identifies and ranks jokes by their relevance to the topic.
    **Important, no duplicate jokes in the database.**
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
    
class ImprovedJokeSchema(BaseModel):
    suggestions: List[str] = Field(
        description="Suggestions to improve the joke",
    )
    new_topic: str = Field(
        description="The new improved topic of the joke, which is generated using the original topic and suggestions",
    )
    
# Create a Pydantic model for the query    
class QuerySchema(BaseModel):
    query: str = Field(
        description="The generated query to find the closest jokes to the given topic",
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
    structured_llm = state["LLM_model"].with_structured_output(FunnySchema)
    prompt = FUNNY_LLM_PROMPT.format(topic=state["joke_topic"])
    res = structured_llm.invoke(prompt)

    try:
        state["messages"] += [
            AIMessage(content=f"Generated joke: {res.joke}"),
            AIMessage(content=f"Topic: {res.topic}"),
            AIMessage(content=f"Rating: {res.rating}"),
        ]
        state["generated_joke"] = res.joke
        state["joke_rating"] = res.rating
        print(f"Joke: {res.joke}")
        print(f"Joke rating: {res.rating}")
        return state
    except Exception as e:
        print(f"The LLM model failed with response:\n{e}\nExiting program")
        exit()

def joke_improver_agent(state: AgentState) -> AgentState:
    print(f"\n**Joke Improver Agent**")
    structured_llm = state["LLM_model"].with_structured_output(ImprovedJokeSchema)
    prompt = IMPROVER_LLM_PROMPT.format(topic=state["joke_topic"], joke=state["generated_joke"])
    res = structured_llm.invoke(prompt)

    try:
        state["messages"] += [
            AIMessage(content=f"Joke suggestions for new topic: {res.suggestions}"),
            AIMessage(content=f"New topic for the joke: {res.new_topic}"),
        ]
        state["joke_topic"] = res.new_topic
        state["iteration"] += 1
        print(f"new topic: {res.new_topic}")
        return state
    except Exception as e:
        print(f"The LLM model failed with response:\n{e}\nExiting program")
        exit()

def database_query_agent(state: AgentState) -> AgentState:
    print(f"\n**Database Query Agent**")
    # Use created schema to structure the output
    structured_llm = state["LLM_model"].with_structured_output(QuerySchema)
    prompt = DATABASE_QUERY_LLM_PROMPT.format(topic=state["joke_topic"], 
                                              tables=tables, 
                                              table_descriptions=description_of_tables, 
                                              joke=state["generated_joke"], 
                                              rating=state["joke_rating"])
    # Invoke the LLM with a prompt and get the structured output
    res = structured_llm.invoke(prompt)
    # Store the result in the state
    state["messages"] += [
        AIMessage(content=f"Generated query: {res.query}"),
    ]
    print(f"Generated query: {res.query}")
    
    try:
        results = sql.run_query(res.query)
        print(f"Results: {results}")
    except Exception as e:
        print(f"Error: {e}")
    
    return state
    

# Create a graph with the state
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("joke", joker_agent)
workflow.add_node("database_query", database_query_agent)
workflow.add_node("joke_improver", joke_improver_agent)

# Edges
workflow.add_edge("joke_improver", "joke")
workflow.add_edge("database_query", END)

def is_done(state):
    # Determ next steps after the first run
    if state["joke_rating"] < 5:
        if state["iteration"] > 5:
            return END
        return "joke_improver"
    else:
        return "database_query"

# Add conditional edge
workflow.add_conditional_edges("joke", is_done)

# Set entry point
workflow.set_entry_point("joke")

# Build the graph
graph = workflow.compile()


if use_cohere:
    print("\nRunning graph with Cohere:\n")
    cohere_chat_model = ChatCohere(cohere_api_key=os.getenv("COHERE_API_KEY"))
    res = graph.invoke({"messages": [HumanMessage(content="Very bad joke about bengal cats")], 
                        "joke_topic": "Very bad joke about bengal cats", 
                        "iteration": 0,
                        "LLM_model": cohere_chat_model})

    print(f"\n\n{res}")
    print(res["messages"])
    print(res["joke_topic"])
    print(f"\n\n{res["generated_joke"]}")

if use_openai:
    print("\nRunning graph with OpenAI:\n")
    openai_chat_model = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    ) 
    res = graph.invoke({"messages": [HumanMessage(content="Very bad joke about bengal cats")], 
                        "joke_topic": "Very bad joke about bengal cats", 
                        "iteration": 0,
                        "LLM_model": openai_chat_model})

    print(f"\n\n{res}")
    print(res["messages"])
    print(res["joke_topic"])
    print(f"\n\n{res["generated_joke"]}")
