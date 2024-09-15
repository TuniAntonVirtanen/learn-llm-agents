'''
    This is a basic example of using Chainlit to create a chatbot
    Run this script with the command: chainlit run 7_chainlit_chat_ui.py
    Main points:
    - Chainlit is used to manage the chatbot's interaction with the user
    - The bot generates jokes based on a user-provided topic
    - A schema is added to structure the output from the language model, including the joke and its rating
    - The LLM output is sent back to the user via Chainlit's messaging interface
'''


import os
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# Select which models you want to use. Cohere = True, OpenAI = False
use_cohere = False


FUNNY_LLM_PROMPT = ChatPromptTemplate.from_template(
    """
    You are the funniest person in the world, a comedian, a joker. You make up jokes about every topic.
    Topic: {topic}                                                      
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


# CHAINLIT - first message when chat starts
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="Hello! I am a funny chatbot. I can make jokes about any topic. What topic would you like me to make a joke about?"
    ).send()


# chainlit - send the joke to the user
@cl.on_message
async def main(message: cl.Message):
    structured_llm = None
    if use_cohere:
        cohere_chat_model = ChatCohere(cohere_api_key=os.getenv("COHERE_API_KEY"))
        structured_llm = cohere_chat_model.with_structured_output(FunnySchema)
    else:
        openai_chat_model = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        )
        structured_llm = openai_chat_model.with_structured_output(FunnySchema)

    prompt = FUNNY_LLM_PROMPT.format(topic=message.content)
    # Invoke the LLM with a prompt and get the structured output
    res = await structured_llm.ainvoke(prompt)
    if res == None:
        await cl.Message(f"Model failed to generate response").send()    
    else:
        await cl.Message(f"Here is a joke about: {res.topic}").send()
        await cl.Message(res.joke).send()
