import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is not set. Check your .env file."
    )


# ============================================================
# 2. GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=api_key,
    temperature=0.4
)


# ============================================================
# 3. RESEARCH PROMPT
# ============================================================

research_prompt = ChatPromptTemplate.from_template("""
You are a web researcher.

Research the following topic:

{topic}

Give me 5 important facts or findings.

Keep the information clear and concise.
""")


# ============================================================
# 4. ANALYSIS PROMPT
# ============================================================

analysis_prompt = ChatPromptTemplate.from_template("""
You are an expert data analyst.

Below is research about:

{topic}

Research:
{research}

Extract 4-6 important insights.

Do not simply repeat the facts.
Identify important trends and connections.
""")


# ============================================================
# 5. REPORT WRITER PROMPT
# ============================================================

writer_prompt = ChatPromptTemplate.from_template("""
You are a professional technical writer.

Create a short markdown report about:

{topic}

Research:
{research}

Insights:
{insights}

Format:

# {topic}

## Summary
2-3 sentences.

## Key Points
- Point 1
- Point 2
- Point 3

## Conclusion
Short conclusion.
""")


# ============================================================
# 6. CREATE CHAINS
# ============================================================

research_chain = research_prompt | llm
analysis_chain = analysis_prompt | llm
writer_chain = writer_prompt | llm


# ============================================================
# 7. PDF GENERATION FUNCTION
# ============================================================

def create_pdf(topic, research, analysis, report):

    # Create output directory
    os.makedirs("output", exist_ok=True)

    pdf_path = "output/final_report.pdf"

    # Create PDF
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    heading_style = styles["Heading1"]
    subheading_style = styles["Heading2"]
    normal_style = styles["BodyText"]

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            f"AI Research Report: {topic}",
            title_style
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Generated using LangChain + Groq",
            normal_style
        )
    )

    story.append(Spacer(1, 30))


    # --------------------------------------------------------
    # RESEARCH
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Research Findings",
            heading_style
        )
    )

    story.append(Spacer(1, 10))

    research_lines = research.split("\n")

    for line in research_lines:

        line = line.strip()

        if not line:
            continue

        # Remove markdown formatting
        line = line.replace("**", "")

        story.append(
            Paragraph(
                line,
                normal_style
            )
        )

        story.append(Spacer(1, 5))


    story.append(PageBreak())


    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Analysis & Insights",
            heading_style
        )
    )

    story.append(Spacer(1, 10))

    analysis_lines = analysis.split("\n")

    for line in analysis_lines:

        line = line.strip()

        if not line:
            continue

        line = line.replace("**", "")

        story.append(
            Paragraph(
                line,
                normal_style
            )
        )

        story.append(Spacer(1, 5))


    story.append(PageBreak())


    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Final Report",
            heading_style
        )
    )

    story.append(Spacer(1, 10))

    report_lines = report.split("\n")

    for line in report_lines:

        line = line.strip()

        if not line:
            story.append(Spacer(1, 8))
            continue

        # Markdown headings
        if line.startswith("### "):

            heading = line.replace("### ", "")

            story.append(
                Paragraph(
                    heading,
                    subheading_style
                )
            )

        elif line.startswith("## "):

            heading = line.replace("## ", "")

            story.append(
                Paragraph(
                    heading,
                    heading_style
                )
            )

        elif line.startswith("# "):

            heading = line.replace("# ", "")

            story.append(
                Paragraph(
                    heading,
                    title_style
                )
            )

        elif line.startswith("- "):

            point = line[2:]

            story.append(
                Paragraph(
                    "• " + point,
                    normal_style
                )
            )

        else:

            line = line.replace("**", "")

            story.append(
                Paragraph(
                    line,
                    normal_style
                )
            )

        story.append(Spacer(1, 5))


    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    doc.build(story)

    return pdf_path


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

topic = input("\nEnter a topic: ")


# ------------------------------------------------------------
# RESEARCH
# ------------------------------------------------------------

research_response = research_chain.invoke({
    "topic": topic
})

research_text = research_response.content

print("\n================ RESEARCH ================")
print(research_text)


# ------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------

analysis_response = analysis_chain.invoke({
    "topic": topic,
    "research": research_text
})

analysis_text = analysis_response.content

print("\n================ ANALYSIS ================")
print(analysis_text)


# ------------------------------------------------------------
# FINAL REPORT
# ------------------------------------------------------------

report_response = writer_chain.invoke({
    "topic": topic,
    "research": research_text,
    "insights": analysis_text
})

report_text = report_response.content

print("\n================ FINAL REPORT ================")
print(report_text)


# ============================================================
# 9. SAVE ALL TEXT FILES
# ============================================================

os.makedirs("output", exist_ok=True)


with open("output/research.txt", "w", encoding="utf-8") as f:
    f.write(research_text)


with open("output/analysis.txt", "w", encoding="utf-8") as f:
    f.write(analysis_text)


with open("output/final_report.md", "w", encoding="utf-8") as f:
    f.write(report_text)


# ============================================================
# 10. GENERATE PDF
# ============================================================

pdf_file = create_pdf(
    topic,
    research_text,
    analysis_text,
    report_text
)


# ============================================================
# 11. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("      PIPELINE COMPLETED SUCCESSFULLY")
print("==========================================")

print("\nGenerated files:")

print("1. output/research.txt")
print("2. output/analysis.txt")
print("3. output/final_report.md")
print("4. output/final_report.pdf")

print("\nPDF generated successfully:")
print(pdf_file)