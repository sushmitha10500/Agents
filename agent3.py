

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

# ============================================================
# 1. API KEY
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set")


# ============================================================
# 2. LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=api_key,
    temperature=0.3
)


# ============================================================
# 3. CREATE TOOLS
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    """

    try:
        result = eval(expression)
        return str(result)

    except Exception as e:
        return f"Error: {e}"


@tool
def get_company_info(company: str) -> str:
    """
    Provides basic information about a company.
    """

    companies = {
        "microsoft":
            "Microsoft is a technology company known for Azure, Windows, Microsoft 365 and AI services.",

        "google":
            "Google is a technology company known for Search, Android, Cloud and Gemini AI.",

        "amazon":
            "Amazon is a technology and e-commerce company known for AWS and online retail."
    }

    return companies.get(
        company.lower(),
        "Company information not available."
    )


# ============================================================
# 4. CREATE AGENT
# ============================================================

agent = create_agent(

    model=llm,

    tools=[
        calculator,
        get_company_info
    ],

    system_prompt="""
You are an intelligent AI assistant.

You have access to tools.

Use the calculator tool for mathematical calculations.

Use the company information tool when the user
asks about supported companies.

Think about the user's request and decide whether
you need a tool.

Give a clear final answer.
"""
)


# ============================================================
# 5. RUN AGENT
# ============================================================

while True:

    user_input = input("\nUser: ")

    if user_input.lower() in ["exit", "quit"]:

        print("Goodbye!")

        break

    result = agent.invoke({

        "messages": [

            {
                "role": "user",
                "content": user_input
            }

        ]

    })

    print("\nAI:")

    print(
        result["messages"][-1].content
    )