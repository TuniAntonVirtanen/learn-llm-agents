import os
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

# This example add schema to the LLM output

# .env file is used to store the api key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# Initialize the language model
# use dotnenv to load OPENAI_API_KEY api key
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
)


# CHAINLIT - first message when chat starts
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="Hello! I am a funny chatbot. I can make jokes about any topic. What topic would you like me to make a joke about?"
    ).send()


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


# chainlit - send the joke to the user
@cl.on_message
async def main(message: cl.Message):
    # Use created schema to structure the output
    structured_llm = llm.with_structured_output(FunnySchema)
    prompt = FUNNY_LLM_PROMPT.format(topic=message.content)
    # Invoke the LLM with a prompt and get the structured output
    res = await structured_llm.ainvoke(prompt)
    await cl.Message(f"Here is a joke about: {res.topic}").send()
    await cl.Message(res.joke).send()
