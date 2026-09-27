import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load .env
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Check your .env file.")


# -----------------------------
# 1. Groq LLM
# -----------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=api_key,
    temperature=0.4
)

# -----------------------------
# 2. Research
# -----------------------------
research = ChatPromptTemplate.from_template("""
You are a web researcher.

Research the following topic:

{topic}

Give me 5 important facts or findings.
Keep the information clear and concise.
""")


# -----------------------------
# 3. Analysis
# -----------------------------
analysis = ChatPromptTemplate.from_template("""
You are an expert data analyst.

Below is research about:

{topic}

Research:
{research}

Extract 4-6 important insights.
Do not simply repeat the facts.
Identify important trends and connections.
""")


# -----------------------------
# 4. Report Writer
# -----------------------------
writer = ChatPromptTemplate.from_template("""
You are a professional technical writer.

Create a short markdown report about:

{topic}

Research:
{research}

Insights:
{insights}

Format:

# Topic

## Summary
2-3 sentences.

## Key Points
- Point 1
- Point 2
- Point 3

## Conclusion
Short conclusion.
""")


# -----------------------------
# 5. Create chains
# -----------------------------
research_chain = research | llm
analysis_chain = analysis | llm
writer_chain = writer | llm


# -----------------------------
# 6. Run pipeline
# -----------------------------
topic = input("Enter a topic: ")

research = research_chain.invoke({
    "topic": topic
})

print("\n===== RESEARCH =====")
print(research.content)


analysis = analysis_chain.invoke({
    "topic": topic,
    "research": research.content
})

print("\n===== ANALYSIS =====")
print(analysis.content)


report = writer_chain.invoke({
    "topic": topic,
    "research": research.content,
    "insights": analysis.content
})

print("\n===== FINAL REPORT =====")
print(report.content)