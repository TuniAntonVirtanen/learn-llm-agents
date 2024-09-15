# This repository uses https://github.com/JoniHonkanen/learn-llm-agents/tree/main as it's basis and is an attempt to enhance it. 
## Main points are:
1. Examine LLM models further
2. Use Cohere alongside OpenAI for comparison and testing purposes
## Summary:
1. This repository expands original with conversation chains, simpler agents and multischema approach
2. Cohere is incapable of managing plethora of tasks that OpenAI model is able to.
Additional files in contrast to the original repository are named with subindexes (ie. 1.x, 3.x...)

# Examples to learn LLM, schemas, agents
1. Create .env file and add there openai and cohere keys
Cohere: https://dashboard.cohere.com/api-keys free tier API-Key is generated automatically when signed in
   -> OPENAI_API_KEY = xxx-xxx-xxx-xxx-xxx
   -> COHERE_API_KEY = xxx-xxx-xxx-xxx-xxx
2. Crate virtualenvironment, use it and install packages
   1. python -m venv .venv
   2. .venv\Scripts\activate
   3. pip install python-dotenv langchain langchain-community langgraph langchain-openai chainlit cohere
3. run program
   ->  python {filename}.py
4. run chainlit (chat ui) (example 7 & 8)
   -> chainlit run {filename}.py
