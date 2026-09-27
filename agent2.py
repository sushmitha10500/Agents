import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import create_agent

load_dotenv()


# Check keys
groq_key = os.getenv("GROQ_API_KEY")
serp_key = os.getenv("SERPAPI_API_KEY")

if not groq_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

if not serp_key:
    raise ValueError("SERPAPI_API_KEY is missing from .env")


# LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    groq_api_key=groq_key
)


# Search
search = SerpAPIWrapper(
    serpapi_api_key=serp_key
)


# Tool
@tool
def google_search(query: str) -> str:
    """Search Google using SerpAPI for current information."""
    return search.run(query)


# Agent
agent = create_agent(
    model=llm,
    tools=[google_search],
    system_prompt="""
You are a research agent.

Find new and recent AI agents.

Use Google search when current information is needed.

For each result explain:
- What it does
- technology used
- Important features
- Source

Never invent information.
"""
)


# User input
question = input("\nWhat would you like to find? ")

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": question
        }
    ]
})


print("\n==============================")
print("FINAL ANSWER")
print("==============================\n")

print(result["messages"][-1].content)