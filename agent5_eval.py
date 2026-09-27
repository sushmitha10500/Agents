import os
import json
import time
from datetime import datetime

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. Check your .env file."
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
# 3. TOOLS
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    """

    try:
        return str(eval(expression))

    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_company_info(company: str) -> str:
    """
    Get basic company information.
    """

    companies = {

        "microsoft": """
Microsoft is a global technology company.
It develops Windows, Microsoft 365, Azure,
Xbox, GitHub and AI services.
""",

        "google": """
Google is a technology company known for
Search, Android, Google Cloud and Gemini.
""",

        "amazon": """
Amazon is a technology and e-commerce company.
AWS is its cloud computing platform.
"""
    }

    return companies.get(
        company.lower(),
        "Information not available."
    )


# ============================================================
# 4. MEMORY
# ============================================================

memory = InMemorySaver()


# ============================================================
# 5. AGENT
# ============================================================

agent = create_agent(

    model=llm,

    tools=[
        calculator,
        get_company_info
    ],

    system_prompt="""
You are an intelligent AI assistant with conversation memory.

MEMORY RULES:

1. Remember important information provided by the user.

2. Maintain continuity across the conversation.

3. Understand follow-up questions.

Example:

User:
Tell me about Microsoft.

User:
How many employees does it have?

Understand "it" as Microsoft.

4. If the user asks about something discussed earlier,
use the conversation history.

5. Never invent memories.

6. If the information is not available in the conversation,
say that you do not know.

7. Use tools when appropriate.

8. Give clear and natural answers.
""",

    checkpointer=memory
)


# ============================================================
# 6. THREAD ID
# ============================================================

THREAD_ID = "sushmita-session-001"

config = {
    "configurable": {
        "thread_id": THREAD_ID
    }
}


# ============================================================
# 7. GET COMPLETE MEMORY
# ============================================================

def get_history():

    state = agent.get_state(config)

    messages = state.values.get(
        "messages",
        []
    )

    history = []

    for message in messages:

        role = message.type

        content = message.content

        if isinstance(content, str):

            history.append(
                f"{role.upper()}: {content}"
            )

    return "\n".join(history)


# ============================================================
# 8. EVALUATE COMPLETE CONVERSATION
# ============================================================

def evaluate_conversation(history):

    evaluation_prompt = f"""
You are a professional AI Agent Evaluator.

You are evaluating an entire conversation between
a USER and an AI AGENT.

Do NOT evaluate only the last message.

Evaluate the complete conversation.

================ CONVERSATION ================

{history}

================================================


Evaluate the agent on these dimensions:

1. Overall Correctness
2. Relevance
3. Completeness
4. Clarity
5. Memory Accuracy
6. Context Understanding
7. Conversation Continuity
8. Tool Usage
9. Instruction Following
10. Hallucination Safety


SCORING:

1 = Very Poor
2 = Poor
3 = Average
4 = Good
5 = Excellent


IMPORTANT:

Memory Accuracy:
Did the agent correctly remember information
from previous messages?

Context Understanding:
Did the agent understand words such as:

"it"
"that"
"this"
"they"
"previously"
"my project"
"the company"

Conversation Continuity:
Did the conversation feel continuous and coherent?

Tool Usage:
Did the agent use the right tool when necessary?

Hallucination Safety:
Did the agent avoid inventing information?


Also identify:

- strengths
- weaknesses
- memory performance
- continuity performance
- tool performance
- improvement suggestions


Return ONLY valid JSON.

Use exactly this structure:

{{
    "overall_correctness": 0,
    "relevance": 0,
    "completeness": 0,
    "clarity": 0,
    "memory_accuracy": 0,
    "context_understanding": 0,
    "conversation_continuity": 0,
    "tool_usage": 0,
    "instruction_following": 0,
    "hallucination_safety": 0,

    "overall_score": 0,

    "strengths": [],

    "weaknesses": [],

    "memory_analysis": "",

    "continuity_analysis": "",

    "tool_analysis": "",

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

    evaluation_text = response.content


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        evaluation = json.loads(
            evaluation_text
        )

    except json.JSONDecodeError:

        evaluation = {

            "error":
                "Evaluator did not return valid JSON.",

            "raw_response":
                evaluation_text
        }


    evaluation[
        "evaluation_time_seconds"
    ] = round(
        evaluation_time,
        2
    )

    return evaluation


# ============================================================
# 9. DISPLAY EVALUATION
# ============================================================

def display_evaluation(evaluation):

    print()
    print("=" * 70)
    print("                 FULL AGENT EVALUATION")
    print("=" * 70)

    if "error" in evaluation:

        print(
            evaluation["raw_response"]
        )

        return


    metrics = [

        ("Overall Correctness",
         "overall_correctness"),

        ("Relevance",
         "relevance"),

        ("Completeness",
         "completeness"),

        ("Clarity",
         "clarity"),

        ("Memory Accuracy",
         "memory_accuracy"),

        ("Context Understanding",
         "context_understanding"),

        ("Conversation Continuity",
         "conversation_continuity"),

        ("Tool Usage",
         "tool_usage"),

        ("Instruction Following",
         "instruction_following"),

        ("Hallucination Safety",
         "hallucination_safety")
    ]


    for label, key in metrics:

        print(
            f"{label:<32}: "
            f"{evaluation.get(key, 'N/A')}/5"
        )


    print("-" * 70)

    print(
        f"{'OVERALL AGENT SCORE':<32}: "
        f"{evaluation.get('overall_score', 'N/A')}/5"
    )

    print(
        f"{'EVALUATION TIME':<32}: "
        f"{evaluation.get('evaluation_time_seconds', 'N/A')} sec"
    )


    # ========================================================
    # STRENGTHS
    # ========================================================

    print()
    print("STRENGTHS")
    print("-" * 70)

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
    print("-" * 70)

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
    print("-" * 70)

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
    print("-" * 70)

    print(
        evaluation.get(
            "continuity_analysis",
            ""
        )
    )


    # ========================================================
    # TOOL
    # ========================================================

    print()
    print("TOOL ANALYSIS")
    print("-" * 70)

    print(
        evaluation.get(
            "tool_analysis",
            ""
        )
    )


    # ========================================================
    # IMPROVEMENTS
    # ========================================================

    print()
    print("IMPROVEMENT SUGGESTIONS")
    print("-" * 70)

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
    print("-" * 70)

    print(
        evaluation.get(
            "final_summary",
            ""
        )
    )

    print("=" * 70)


# ============================================================
# 10. SAVE EVALUATION
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

    file_path = (
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
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    return file_path


# ============================================================
# 11. MAIN CHAT
# ============================================================

print()
print("=" * 70)
print("        AGENT + MEMORY + CONTINUITY + EVALUATION")
print("=" * 70)

print()
print("Thread ID:", THREAD_ID)

print()
print("Commands:")
print("  history  → Show complete conversation")
print("  evaluate → Evaluate complete conversation")
print("  exit     → Exit")
print()


while True:

    user_input = input("User: ").strip()


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
        print("=" * 70)
        print("                  CONVERSATION MEMORY")
        print("=" * 70)

        if history:

            print(history)

        else:

            print(
                "No conversation yet."
            )

        print("=" * 70)

        continue


    # ========================================================
    # EVALUATE
    # ========================================================

    if user_input.lower() == "evaluate":

        history = get_history()


        if not history:

            print(
                "\nNo conversation available for evaluation."
            )

            continue


        print()
        print(
            "Evaluating complete conversation..."
        )


        evaluation = evaluate_conversation(
            history
        )


        display_evaluation(
            evaluation
        )


        evaluation_file = save_evaluation(

            history,

            evaluation
        )


        print()
        print(
            "Evaluation saved to:"
        )

        print(
            evaluation_file
        )

        continue


    # ========================================================
    # NORMAL AGENT CHAT
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