import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is not set. Check your .env file."
    )


# ============================================================
# 2. CREATE LLM
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

        return f"Calculation error: {e}"


@tool
def get_company_info(company: str) -> str:
    """
    Provides basic information about a company.
    """

    companies = {

        "microsoft":
            """
            Microsoft is a global technology company.
            It develops Windows, Microsoft 365, Azure,
            Xbox, Surface devices, GitHub and AI services.
            """,

        "google":
            """
            Google is a technology company known for
            Search, Android, Google Cloud and Gemini AI.
            """,

        "amazon":
            """
            Amazon is a technology and e-commerce company.
            AWS is its major cloud computing platform.
            """
    }

    return companies.get(
        company.lower(),
        "Company information is not available."
    )


# ============================================================
# 4. MEMORY
# ============================================================

memory = InMemorySaver()


# ============================================================
# 5. CREATE AGENT
# ============================================================

agent = create_agent(

    model=llm,

    tools=[
        calculator,
        get_company_info
    ],

    system_prompt="""
You are an intelligent AI assistant with conversation memory.

Remember important information from the conversation
and use previous messages when answering follow-up questions.

For example:

User:
Tell me about Microsoft.

User:
How many employees does it have?

Understand that "it" refers to Microsoft.

If the user asks a follow-up question, use the
previous conversation to understand the context.

Use the calculator tool for mathematical calculations.

Use the company information tool when appropriate.

Give clear and concise answers.
""",

    checkpointer=memory
)


# ============================================================
# 6. THREAD ID
# ============================================================

config = {
    "configurable": {
        "thread_id": "user-1"
    }
}


# ============================================================
# 7. CHAT LOOP
# ============================================================

print("\n==========================================")
print("      AI AGENT WITH MEMORY")
print("==========================================")
print("Type 'exit' to stop.")
print("")


while True:

    user_input = input("User: ")

    if user_input.lower() in ["exit", "quit"]:

        print("\nGoodbye!")

        break


    # --------------------------------------------------------
    # INVOKE AGENT
    # --------------------------------------------------------

    result = agent.invoke(

        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        },

        config=config
    )


    # --------------------------------------------------------
    # GET LAST MESSAGE
    # --------------------------------------------------------

    print("\nAI:")

    print(
        result["messages"][-1].content
    )

    print()