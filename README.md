# Examples to learn LLM, schemas, agents

1. Create .env file and add there openai key
   -> OPENAI_API_KEY = xxx-xxx-xxx-xxx-xxx
2. Crate virtualenvironment, use it and install packages
   1. python -m venv .venv
   2. .venv\Scripts\activate
   3. pip install python-dotenv langchain langchain-community langgraph langchain-openai chainlit
3. run program
   ->  python {filename}.py
4. run chainlit (chat ui) (example 7 & 8)
   -> chainlit run {filename}.py
