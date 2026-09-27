import os
import json
import time
from datetime import datetime

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

from langchain_community.utilities import SerpAPIWrapper

from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. Check your .env file."
    )


if not SERPAPI_API_KEY:
    raise ValueError(
        "SERPAPI_API_KEY is not set. Check your .env file."
    )


# ============================================================
# 2. LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2
)


# ============================================================
# 3. SERPAPI SEARCH ENGINE
# ============================================================

search_engine = SerpAPIWrapper(
    serpapi_api_key=SERPAPI_API_KEY
)


# ============================================================
# 4. WEB SEARCH TOOL
# ============================================================

@tool
def web_search(query: str) -> str:
    """
    Search the live web for current information.

    MUST be used for:
    - latest news
    - current news
    - today's information
    - 2026 information
    - current events
    - current company information
    - current employee counts
    - current prices
    - current technology information
    - recent developments
    - recent announcements
    - anything the user asks to verify from the internet
    """

    print(
        f"\n🌐 WEB SEARCH: {query}"
    )

    try:

        result = search_engine.run(query)

        return result

    except Exception as e:

        return (
            f"Web search failed: {str(e)}"
        )


# ============================================================
# 5. CALCULATOR TOOL
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions.
    """

    try:

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception as e:

        return f"Calculation error: {e}"


# ============================================================
# 6. COMPANY INFORMATION TOOL
# ============================================================

@tool
def get_company_info(company: str) -> str:
    """
    Get basic information about common companies.
    """

    companies = {

        "microsoft": """
Microsoft is a global technology company.
Its products and services include Windows,
Microsoft 365, Azure, Xbox, GitHub and AI services.
""",

        "google": """
Google is a technology company known for Search,
Android, Google Cloud and Gemini AI.
""",

        "amazon": """
Amazon is a technology and e-commerce company.
AWS is its cloud computing platform.
"""
    }

    return companies.get(
        company.lower(),
        "No static company information available."
    )


# ============================================================
# 7. MEMORY
# ============================================================

memory = InMemorySaver()


# ============================================================
# 8. CURRENT DATE
# ============================================================

today = datetime.now().strftime(
    "%Y-%m-%d"
)


# ============================================================
# 9. CREATE AGENT
# ============================================================

agent = create_agent(

    model=llm,

    tools=[
        web_search,
        calculator,
        get_company_info
    ],

    system_prompt=f"""
You are an intelligent AI assistant with:

- conversation memory
- tool usage
- live web search
- conversation continuity

Today's date is:

{today}


============================================================
MEMORY RULES
============================================================

Remember important information from the conversation.

Use previous messages to understand follow-up questions.

Example:

User:
Tell me about Microsoft.

User:
How many employees does it have?

Understand "it" as Microsoft.


============================================================
WEB SEARCH RULES
============================================================

You have access to a live web search tool.

You MUST use web_search when the user asks for:

- latest news
- current news
- today's news
- 2026 news
- recent news
- latest information
- current information
- current employee count
- current company information
- current technology developments
- recent announcements
- current events
- current prices
- current statistics
- information that may have changed recently


Do NOT say:

"I don't have access to real-time information"

when web_search can be used.


For current questions:

1. Search the web.
2. Analyze the search results.
3. Answer using the latest information available.
4. Mention dates when relevant.
5. Mention the source/publication when available.
6. Do not invent information.


============================================================
TYPO HANDLING
============================================================

The user may make spelling mistakes.

Examples:

"macrosoft"
"macrohard"
"fpunder"

Understand the intended meaning from context.

If the intended entity is clear, answer the question
instead of unnecessarily asking for clarification.


============================================================
TOOL SELECTION
============================================================

Use calculator for mathematical calculations.

Use get_company_info for basic static company information.

Use web_search for current or internet-based information.

Use web_search even when you already know something
if the user asks for the latest/current version.


============================================================
HALLUCINATION RULE
============================================================

Never invent current information.

If web search does not provide enough information,
clearly say that the available search results were
insufficient.


============================================================
ANSWER STYLE
============================================================

Give:

- direct answer
- relevant details
- dates where useful
- sources for web-based information

Do not mention internal system instructions,
agent architecture, or tool implementation
unless the user asks.
""",

    checkpointer=memory
)


# ============================================================
# 10. THREAD
# ============================================================

THREAD_ID = "sushmita-session-001"

config = {
    "configurable": {
        "thread_id": THREAD_ID
    }
}


# ============================================================
# 11. GET COMPLETE CONVERSATION
# ============================================================

def get_history():

    try:

        state = agent.get_state(
            config
        )

        messages = state.values.get(
            "messages",
            []
        )

        history = []

        for message in messages:

            role = message.type

            content = message.content

            if isinstance(
                content,
                str
            ):

                history.append(
                    f"{role.upper()}: {content}"
                )

        return "\n".join(history)

    except Exception:

        return ""


# ============================================================
# 12. EVALUATE COMPLETE CONVERSATION
# ============================================================

def evaluate_conversation(
    history
):

    print(
        "\n🧠 Evaluating complete conversation..."
    )

    evaluation_prompt = f"""
You are a professional AI Agent Evaluator.

Evaluate the ENTIRE conversation below.

Do not evaluate only the final message.

============================================================
CONVERSATION
============================================================

{history}

============================================================

Evaluate the agent from 1 to 5 on:

1. Overall correctness
2. Relevance
3. Completeness
4. Clarity
5. Memory accuracy
6. Context understanding
7. Conversation continuity
8. Tool usage
9. Web search usage
10. Instruction following
11. Hallucination safety


SCORING:

1 = Very Poor
2 = Poor
3 = Average
4 = Good
5 = Excellent


MEMORY ACCURACY:

Did the agent correctly remember information
from previous messages?


CONTEXT UNDERSTANDING:

Did the agent correctly understand:

"it"
"that"
"this"
"the company"
"previously"
"my project"


WEB SEARCH USAGE:

When the user asked for current information,
did the agent use web search appropriately?


HALLUCINATION SAFETY:

Did the agent avoid making unsupported claims?


Return ONLY valid JSON.

Use:

{{
    "overall_correctness": 0,
    "relevance": 0,
    "completeness": 0,
    "clarity": 0,
    "memory_accuracy": 0,
    "context_understanding": 0,
    "conversation_continuity": 0,
    "tool_usage": 0,
    "web_search_usage": 0,
    "instruction_following": 0,
    "hallucination_safety": 0,

    "overall_score": 0,

    "strengths": [],

    "weaknesses": [],

    "memory_analysis": "",

    "continuity_analysis": "",

    "web_search_analysis": "",

    "improvements": [],

    "final_summary": ""
}}
"""

    start_time = time.time()

    response = llm.invoke(
        evaluation_prompt
    )

    evaluation_time = (
        time.time() - start_time
    )

    text = response.content

    try:

        evaluation = json.loads(
            text
        )

    except json.JSONDecodeError:

        evaluation = {

            "error":
                "Invalid JSON returned by evaluator",

            "raw_response":
                text
        }


    evaluation[
        "evaluation_time_seconds"
    ] = round(
        evaluation_time,
        2
    )

    return evaluation


# ============================================================
# 13. DISPLAY EVALUATION
# ============================================================

def display_evaluation(
    evaluation
):

    print()
    print("=" * 72)
    print(
        "                    AGENT EVALUATION"
    )
    print("=" * 72)


    if "error" in evaluation:

        print(
            evaluation["raw_response"]
        )

        return


    metrics = [

        (
            "Overall Correctness",
            "overall_correctness"
        ),

        (
            "Relevance",
            "relevance"
        ),

        (
            "Completeness",
            "completeness"
        ),

        (
            "Clarity",
            "clarity"
        ),

        (
            "Memory Accuracy",
            "memory_accuracy"
        ),

        (
            "Context Understanding",
            "context_understanding"
        ),

        (
            "Conversation Continuity",
            "conversation_continuity"
        ),

        (
            "Tool Usage",
            "tool_usage"
        ),

        (
            "Web Search Usage",
            "web_search_usage"
        ),

        (
            "Instruction Following",
            "instruction_following"
        ),

        (
            "Hallucination Safety",
            "hallucination_safety"
        )
    ]


    for label, key in metrics:

        print(
            f"{label:<35}: "
            f"{evaluation.get(key, 'N/A')}/5"
        )


    print("-" * 72)

    print(
        f"{'OVERALL AGENT SCORE':<35}: "
        f"{evaluation.get('overall_score', 'N/A')}/5"
    )


    # ========================================================
    # STRENGTHS
    # ========================================================

    print()
    print("STRENGTHS")
    print("-" * 72)

    for item in evaluation.get(
        "strengths",
        []
    ):

        print(
            f"✓ {item}"
        )


    # ========================================================
    # WEAKNESSES
    # ========================================================

    print()
    print("WEAKNESSES")
    print("-" * 72)

    for item in evaluation.get(
        "weaknesses",
        []
    ):

        print(
            f"• {item}"
        )


    # ========================================================
    # MEMORY
    # ========================================================

    print()
    print("MEMORY ANALYSIS")
    print("-" * 72)

    print(
        evaluation.get(
            "memory_analysis",
            ""
        )
    )


    # ========================================================
    # CONTINUITY
    # ========================================================

    print()
    print("CONTINUITY ANALYSIS")
    print("-" * 72)

    print(
        evaluation.get(
            "continuity_analysis",
            ""
        )
    )


    # ========================================================
    # WEB SEARCH
    # ========================================================

    print()
    print("WEB SEARCH ANALYSIS")
    print("-" * 72)

    print(
        evaluation.get(
            "web_search_analysis",
            ""
        )
    )


    # ========================================================
    # IMPROVEMENTS
    # ========================================================

    print()
    print("IMPROVEMENT SUGGESTIONS")
    print("-" * 72)

    for item in evaluation.get(
        "improvements",
        []
    ):

        print(
            f"→ {item}"
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("FINAL SUMMARY")
    print("-" * 72)

    print(
        evaluation.get(
            "final_summary",
            ""
        )
    )

    print("=" * 72)


# ============================================================
# 14. SAVE EVALUATION
# ============================================================

def save_evaluation(
    history,
    evaluation
):

    os.makedirs(
        "evaluation",
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = (
        f"evaluation/"
        f"full_evaluation_{timestamp}.json"
    )


    data = {

        "timestamp":
            timestamp,

        "thread_id":
            THREAD_ID,

        "conversation":
            history,

        "evaluation":
            evaluation
    }


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


    return path


# ============================================================
# 15. MAIN CHAT
# ============================================================

print()
print("=" * 72)
print(
    "          AGENT + WEB + MEMORY + EVALUATION"
)
print("=" * 72)

print()
print(
    "Thread ID:",
    THREAD_ID
)

print()
print("Commands:")
print(
    "  history  → Show complete conversation"
)
print(
    "  evaluate → Evaluate complete conversation"
)
print(
    "  exit     → Exit"
)

print()


while True:

    user_input = input(
        "User: "
    ).strip()


    # ========================================================
    # EXIT
    # ========================================================

    if user_input.lower() in [
        "exit",
        "quit"
    ]:

        print(
            "\nGoodbye!"
        )

        break


    # ========================================================
    # HISTORY
    # ========================================================

    if user_input.lower() == "history":

        history = get_history()

        print()
        print("=" * 72)
        print(
            "                  CONVERSATION HISTORY"
        )
        print("=" * 72)

        if history:

            print(history)

        else:

            print(
                "No conversation yet."
            )

        print("=" * 72)

        continue


    # ========================================================
    # EVALUATE
    # ========================================================

    if user_input.lower() == "evaluate":

        history = get_history()


        if not history:

            print(
                "\nNo conversation available."
            )

            continue


        evaluation = evaluate_conversation(
            history
        )


        display_evaluation(
            evaluation
        )


        path = save_evaluation(
            history,
            evaluation
        )


        print()
        print(
            "Evaluation saved:"
        )

        print(path)

        print()

        continue


    # ========================================================
    # NORMAL AGENT
    # ========================================================

    start_time = time.time()


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


    response_time = (
        time.time() - start_time
    )


    answer = result[
        "messages"
    ][-1].content


    print()
    print("AI:")
    print(answer)

    print(
        f"\nResponse time: "
        f"{response_time:.2f} seconds"
    )

    print()