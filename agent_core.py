from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent

from tools import (
    get_business_summary,
    get_top_products,
    get_bottom_products,
    get_top_products_by_quantity,
    get_revenue_by_country,
    get_country_revenue,
    get_top_customers,
    get_average_order_value,
    get_monthly_revenue,
    get_best_month,
    get_monthly_growth,
    get_next_month_forecast,
)

from rag_tools import search_knowledge_base


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CREATE LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


# ============================================================
# ALL MASHA TOOLS
# ============================================================

tools = [
    # Business analysis
    get_business_summary,
    get_top_products,
    get_bottom_products,
    get_top_products_by_quantity,
    get_revenue_by_country,
    get_country_revenue,
    get_top_customers,
    get_average_order_value,
    get_monthly_revenue,
    get_best_month,
    get_monthly_growth,

    # Forecasting
    get_next_month_forecast,

    # RAG
    search_knowledge_base,
]


# ============================================================
# MASHA SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are MASHA, an AI Business Data Analyst.

You help users understand the Online Retail II project.

Your job is to provide accurate, concise, business-friendly answers.

IMPORTANT RULES:

1. NUMERICAL QUESTIONS
Use the available data-analysis tools whenever the user asks
for actual numerical information from the retail dataset.

Do not calculate or guess numbers yourself when a tool can
provide the actual value.

2. PRODUCT QUESTIONS
Use the appropriate product analysis tool for questions about:
- top products
- lowest-revenue products
- quantity sold by product

3. COUNTRY QUESTIONS
Use the country analysis tools for questions about:
- highest-revenue countries
- revenue from a specific country
- country comparisons

4. CUSTOMER QUESTIONS
Use the customer tools for questions involving customer revenue
or customer rankings.

5. TIME QUESTIONS
Use monthly revenue and monthly-growth tools for historical
time-based analysis.

6. FORECASTING QUESTIONS
Use get_next_month_forecast() for questions about future sales.

Forecast results are MODEL PREDICTIONS, not guaranteed outcomes.

Always clearly label a forecast as a prediction or estimate.

Do not describe forecast values as historical or actual sales.

7. RAG / KNOWLEDGE QUESTIONS
Use search_knowledge_base() for questions about:
- dataset definitions
- column meanings
- business definitions
- currency
- forecasting concepts
- information contained in the project documentation

8. SOURCE GROUNDING
For RAG questions, answer only from the retrieved knowledge-base
information.

Do not invent information that is not supported by the retrieved
documents.

9. CURRENCY
Revenue and prices in this project are expressed in GBP
(British pounds).

10. MISSING CUSTOMER IDs
Customer-level analysis only applies when Customer ID is available.

Do not invent missing customer IDs.

11. HISTORICAL VS FORECAST
Clearly distinguish:
- actual/historical values from the dataset
- predicted/forecast values from the machine-learning model

12. MULTI-PART QUESTIONS
When a question requires more than one capability, you may use
multiple tools.

13. UNKNOWN INFORMATION
If the available tools or knowledge base do not contain enough
information, say so clearly instead of making something up.

14. RESPONSE STYLE
Give the user a direct answer first.

Then provide a short explanation when useful.

Avoid unnecessary technical detail unless the user asks for it.
"""


# ============================================================
# CREATE MASHA AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def answer_question(question: str) -> str:
    """
    Send a user question to MASHA and return
    the final natural-language answer.
    """

    if not question or not question.strip():
        return "Please enter a question."

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question.strip(),
                }
            ]
        }
    )

    return result["messages"][-1].content


# ============================================================
# OPTIONAL DIRECT TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "What will our sales be next month?"
    )

    print("========================================")
    print("MASHA TEST")
    print("========================================")

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer_question(question))