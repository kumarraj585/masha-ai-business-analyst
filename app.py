import streamlit as st
import pandas as pd

from agent_core import answer_question

from tools import (
    get_business_summary,
    get_average_order_value,
    get_top_products,
    get_revenue_by_country,
    get_next_month_forecast,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MASHA | AI Business Analyst",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
"""
<style>

/* ============================================================
   GLOBAL BACKGROUND
   ============================================================ */

.stApp {
    min-height: 100vh;

    background:
        radial-gradient(
            circle at 8% 12%,
            rgba(124, 58, 237, 0.22),
            transparent 25%
        ),
        radial-gradient(
            circle at 92% 10%,
            rgba(37, 99, 235, 0.18),
            transparent 24%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(14, 116, 144, 0.12),
            transparent 28%
        ),
        linear-gradient(
            135deg,
            #050816 0%,
            #0a1020 50%,
            #070b15 100%
        );

    color: #f8fafc;
}

.block-container {
    max-width: 1100px;
    padding-top: 28px;
    padding-bottom: 60px;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* ============================================================
   HEADER
   ============================================================ */

.masha-header {
    text-align: center;
    margin-bottom: 28px;
}

.masha-icon {
    width: 58px;
    height: 58px;

    margin: 0 auto 12px auto;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 18px;

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #2563eb
        );

    color: white;

    font-size: 25px;
    font-weight: 800;

    box-shadow:
        0 0 35px rgba(124, 58, 237, 0.40);
}

.masha-title {
    color: #ffffff !important;

    font-size: 36px;
    font-weight: 850;

    letter-spacing: -0.04em;
}

.masha-subtitle {
    color: #94a3b8 !important;

    font-size: 14px;
    margin-top: 4px;
}


/* ============================================================
   ASK AREA
   ============================================================ */

.ask-card {
    max-width: 800px;

    margin: 0 auto;

    padding: 30px;

    background:
        rgba(15, 23, 42, 0.78);

    border:
        1px solid rgba(148, 163, 184, 0.13);

    border-radius: 24px;

    box-shadow:
        0 25px 70px rgba(0, 0, 0, 0.30);

    backdrop-filter: blur(18px);
}

.ask-title {
    text-align: center;

    color: #ffffff !important;

    font-size: 27px;
    font-weight: 800;

    letter-spacing: -0.03em;
}

.ask-description {
    text-align: center;

    color: #94a3b8 !important;

    font-size: 13px;

    line-height: 1.6;

    margin-top: 6px;
}


/* ============================================================
   INPUT
   ============================================================ */

div[data-testid="stTextInput"] input {
    background: #0f172a !important;

    color: #ffffff !important;

    border:
        1px solid rgba(129, 140, 248, 0.25) !important;

    border-radius: 14px !important;

    min-height: 52px;

    font-size: 15px;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.16);
}

div[data-testid="stTextInput"] input:focus {
    border:
        1px solid rgba(129, 140, 248, 0.65) !important;

    box-shadow:
        0 0 0 1px rgba(129, 140, 248, 0.20),
        0 0 28px rgba(99, 102, 241, 0.12);
}

div[data-testid="stTextInput"] input::placeholder {
    color: #64748b !important;
}


/* ============================================================
   ASK BUTTON
   ============================================================ */

.ask-button .stButton > button {
    width: 100%;

    min-height: 50px;

    border: none !important;

    border-radius: 14px !important;

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #2563eb
        ) !important;

    color: #ffffff !important;

    font-size: 15px !important;

    font-weight: 750 !important;

    box-shadow:
        0 10px 30px rgba(99, 102, 241, 0.22);

    transition:
        transform 0.16s ease,
        box-shadow 0.16s ease;
}

.ask-button .stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 16px 38px rgba(99, 102, 241, 0.35);
}


/* ============================================================
   ANSWER
   ============================================================ */

.answer-wrapper {
    max-width: 800px;

    margin: 22px auto 30px auto;
}

.answer-label {
    color: #a78bfa !important;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 0.13em;

    text-transform: uppercase;

    margin-bottom: 8px;
}

.answer-card {
    background:
        linear-gradient(
            145deg,
            rgba(30, 27, 75, 0.74),
            rgba(15, 23, 42, 0.90)
        );

    border:
        1px solid rgba(129, 140, 248, 0.18);

    border-left:
        4px solid #8b5cf6;

    border-radius: 18px;

    padding: 21px;

    color: #f8fafc !important;

    line-height: 1.65;

    box-shadow:
        0 15px 40px rgba(0,0,0,0.25);
}

.answer-card p,
.answer-card li,
.answer-card strong,
.answer-card em {
    color: #f8fafc !important;
}


/* ============================================================
   QUICK QUESTIONS
   ============================================================ */

.quick-label {
    text-align: center;

    color: #64748b !important;

    font-size: 12px;

    margin-top: 14px;
    margin-bottom: 11px;
}

.quick-button .stButton > button {
    background:
        rgba(15, 23, 42, 0.72) !important;

    color: #cbd5e1 !important;

    border:
        1px solid rgba(148, 163, 184, 0.12) !important;

    border-radius: 12px !important;

    min-height: 42px;

    font-weight: 650 !important;

    transition:
        transform 0.15s ease,
        border-color 0.15s ease;
}

.quick-button .stButton > button:hover {
    transform: translateY(-2px);

    border-color:
        rgba(129, 140, 248, 0.40) !important;

    color: #ffffff !important;
}


/* ============================================================
   SECTION
   ============================================================ */

.section-title {
    color: #ffffff !important;

    font-size: 22px;

    font-weight: 780;

    letter-spacing: -0.025em;

    margin-top: 42px;

    margin-bottom: 7px;
}

.section-caption {
    color: #64748b !important;

    font-size: 13px;

    margin-bottom: 16px;
}


/* ============================================================
   KPI CARDS
   ============================================================ */

.kpi {
    background:
        rgba(15, 23, 42, 0.74);

    border:
        1px solid rgba(148, 163, 184, 0.10);

    border-radius: 17px;

    padding: 18px;

    min-height: 108px;

    box-shadow:
        0 10px 28px rgba(0,0,0,0.16);

    backdrop-filter: blur(14px);

    transition:
        transform 0.15s ease,
        border-color 0.15s ease;
}

.kpi:hover {
    transform: translateY(-3px);

    border-color:
        rgba(129, 140, 248, 0.28);
}

.kpi-label {
    color: #64748b !important;

    font-size: 10px;

    font-weight: 750;

    text-transform: uppercase;

    letter-spacing: 0.09em;
}

.kpi-value {
    color: #ffffff !important;

    font-size: 25px;

    font-weight: 820;

    margin-top: 9px;
}


/* ============================================================
   CHART
   ============================================================ */

[data-testid="stLineChart"] {
    background:
        rgba(15, 23, 42, 0.74);

    border:
        1px solid rgba(148, 163, 184, 0.10);

    border-radius: 18px;

    padding: 8px;

    box-shadow:
        0 12px 32px rgba(0,0,0,0.18);
}


/* ============================================================
   INSIGHTS
   ============================================================ */

.insight {
    background:
        rgba(15, 23, 42, 0.74);

    border:
        1px solid rgba(148, 163, 184, 0.10);

    border-radius: 17px;

    padding: 19px;

    min-height: 115px;

    box-shadow:
        0 10px 28px rgba(0,0,0,0.16);
}

.insight-label {
    color: #64748b !important;

    font-size: 10px;

    font-weight: 750;

    text-transform: uppercase;

    letter-spacing: 0.09em;
}

.insight-value {
    color: #ffffff !important;

    font-size: 17px;

    font-weight: 750;

    margin-top: 8px;
}

.insight-note {
    color: #94a3b8 !important;

    font-size: 12px;

    margin-top: 5px;
}


/* ============================================================
   FORECAST BUTTON
   ============================================================ */

.forecast-button .stButton > button {
    width: 100%;

    min-height: 45px;

    border-radius: 13px !important;

    background:
        rgba(124, 58, 237, 0.12) !important;

    color: #c4b5fd !important;

    border:
        1px solid rgba(124, 58, 237, 0.25) !important;

    font-weight: 700 !important;
}

.forecast-button .stButton > button:hover {
    background:
        rgba(124, 58, 237, 0.20) !important;

    border-color:
        rgba(124, 58, 237, 0.45) !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    color: #475569 !important;

    font-size: 11px;

    margin-top: 50px;

    padding-top: 20px;

    border-top:
        1px solid rgba(148, 163, 184, 0.07);
}

</style>
""",
unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_dashboard_data():

    df = pd.read_csv(
        "data/online_retail.csv"
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"]
    )

    df["Revenue"] = (
        df["Quantity"] * df["Price"]
    )

    return df


df = load_dashboard_data()


# ============================================================
# CACHE LIGHTWEIGHT ANALYSIS
# ============================================================

@st.cache_data
def get_summary_cached():
    return get_business_summary.invoke({})


@st.cache_data
def get_aov_cached():
    return get_average_order_value.invoke({})


@st.cache_data
def get_top_products_cached():
    return get_top_products.invoke({"n": 5})


@st.cache_data
def get_countries_cached():
    return get_revenue_by_country.invoke({"n": 5})


# IMPORTANT:
# Forecast is NOT calculated when the page opens.


# ============================================================
# GET FAST DASHBOARD DATA
# ============================================================

summary = get_summary_cached()

aov = get_aov_cached()

top_products = get_top_products_cached()

countries = get_countries_cached()


# ============================================================
# HEADER
# ============================================================

st.markdown(
"""
<div class="masha-header">

<div class="masha-icon">
✦
</div>

<div class="masha-title">
MASHA
</div>

<div class="masha-subtitle">
AI Business Data Analyst
</div>

</div>
""",
unsafe_allow_html=True,
)


# ============================================================
# ASK MASHA
# ============================================================

st.markdown(
"""
<div class="ask-card">

<div class="ask-title">
Ask MASHA
</div>

<div class="ask-description">
Ask questions about your business data,
customers, products, markets, forecasts, or knowledge base.
</div>

</div>
""",
unsafe_allow_html=True,
)


question = st.text_input(
    "Question",

    placeholder=(
        "e.g. Which product generated the most revenue?"
    ),

    label_visibility="collapsed",
)


# ============================================================
# ASK BUTTON
# ============================================================

st.markdown(
'<div class="ask-button">',
unsafe_allow_html=True,
)

ask_clicked = st.button(
    "✦  Ask MASHA",
    use_container_width=True,
)

st.markdown(
'</div>',
unsafe_allow_html=True,
)


# ============================================================
# ASK MASHA
# ============================================================

if ask_clicked:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "MASHA is analyzing..."
        ):

            try:

                answer = answer_question(
                    question
                )

                st.markdown(
                    '<div class="answer-wrapper">',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="answer-label">'
                    '✦ AI ANSWER'
                    '</div>',
                    unsafe_allow_html=True,
                )

                # Use Streamlit markdown rather than
                # placing the model output inside HTML.
                st.markdown(
                    '<div class="answer-card">',
                    unsafe_allow_html=True,
                )

                st.markdown(answer)

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True,
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )


# ============================================================
# QUICK QUESTIONS
# ============================================================

st.markdown(
'<div class="quick-label">Quick questions</div>',
unsafe_allow_html=True,
)

q1, q2, q3, q4 = st.columns(4)


with q1:

    st.markdown(
        '<div class="quick-button">',
        unsafe_allow_html=True,
    )

    if st.button(
        "🏆 Top products",
        use_container_width=True,
    ):

        st.session_state["quick_question"] = (
            "Which 5 products generated the most revenue?"
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


with q2:

    st.markdown(
        '<div class="quick-button">',
        unsafe_allow_html=True,
    )

    if st.button(
        "🌍 Best country",
        use_container_width=True,
    ):

        st.session_state["quick_question"] = (
            "Which country generated the most revenue?"
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


with q3:

    st.markdown(
        '<div class="quick-button">',
        unsafe_allow_html=True,
    )

    if st.button(
        "📦 Top quantity",
        use_container_width=True,
    ):

        st.session_state["quick_question"] = (
            "Which products sold the most units?"
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


with q4:

    st.markdown(
        '<div class="quick-button">',
        unsafe_allow_html=True,
    )

    if st.button(
        "📚 Customer ID",
        use_container_width=True,
    ):

        st.session_state["quick_question"] = (
            "What does Customer ID mean?"
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# QUICK QUESTION
# ============================================================

if st.session_state.get("quick_question"):

    quick_question = st.session_state.pop(
        "quick_question"
    )

    with st.spinner(
        "MASHA is analyzing..."
    ):

        try:

            answer = answer_question(
                quick_question
            )

            st.markdown(
                '<div class="answer-wrapper">',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="answer-label">'
                '✦ AI ANSWER'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="answer-card">',
                unsafe_allow_html=True,
            )

            st.markdown(answer)

            st.markdown(
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True,
            )

        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )


# ============================================================
# BUSINESS SNAPSHOT
# ============================================================

st.markdown(
'<div class="section-title">Business Snapshot</div>',
unsafe_allow_html=True,
)

st.markdown(
'<div class="section-caption">'
'Key metrics from the interactive retail dataset'
'</div>',
unsafe_allow_html=True,
)


k1, k2, k3, k4, k5 = st.columns(5)


with k1:

    st.markdown(
        f"""
<div class="kpi">

<div class="kpi-label">
Revenue
</div>

<div class="kpi-value">
£{summary["total_revenue"]:,.0f}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with k2:

    st.markdown(
        f"""
<div class="kpi">

<div class="kpi-label">
Units Sold
</div>

<div class="kpi-value">
{summary["total_quantity"]:,}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with k3:

    st.markdown(
        f"""
<div class="kpi">

<div class="kpi-label">
Orders
</div>

<div class="kpi-value">
{summary["total_transactions"]:,}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with k4:

    st.markdown(
        f"""
<div class="kpi">

<div class="kpi-label">
Countries
</div>

<div class="kpi-value">
{summary["unique_countries"]}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with k5:

    st.markdown(
        f"""
<div class="kpi">

<div class="kpi-label">
Avg Order
</div>

<div class="kpi-value">
£{aov["average_order_value"]:,.2f}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# MONTHLY SALES
# ============================================================

st.markdown(
'<div class="section-title">Monthly Sales</div>',
unsafe_allow_html=True,
)

st.markdown(
'<div class="section-caption">'
'Revenue trend across the interactive dataset'
'</div>',
unsafe_allow_html=True,
)


monthly = (
    df.groupby(
        df["InvoiceDate"].dt.to_period("M")
    )["Revenue"]
    .sum()
)

monthly.index = monthly.index.astype(str)

st.line_chart(
    monthly,
    height=320,
)


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.markdown(
'<div class="section-title">Business Insights</div>',
unsafe_allow_html=True,
)


i1, i2, i3 = st.columns(3)


with i1:

    product = list(
        top_products["products"].items()
    )[0]

    st.markdown(
        f"""
<div class="insight">

<div class="insight-label">
🏆 Top Product
</div>

<div class="insight-value">
{product[0]}
</div>

<div class="insight-note">
£{product[1]:,.2f} revenue
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with i2:

    country = list(
        countries["countries"].items()
    )[0]

    st.markdown(
        f"""
<div class="insight">

<div class="insight-label">
🌍 Top Market
</div>

<div class="insight-value">
{country[0]}
</div>

<div class="insight-note">
£{country[1]:,.2f} revenue
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with i3:

    st.markdown(
        """
<div class="insight">

<div class="insight-label">
🔮 Forecast
</div>

<div class="insight-value">
Available on demand
</div>

<div class="insight-note">
Random Forest · Daily model
</div>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# ON-DEMAND FORECAST
# ============================================================

st.markdown(
'<div class="section-title">Sales Forecast</div>',
unsafe_allow_html=True,
)

st.markdown(
'<div class="section-caption">'
'The forecast is calculated only when requested, keeping MASHA fast.'
'</div>',
unsafe_allow_html=True,
)


st.markdown(
'<div class="forecast-button">',
unsafe_allow_html=True,
)

forecast_clicked = st.button(
    "🔮 Generate Next-Month Forecast",
    use_container_width=True,
)

st.markdown(
'</div>',
unsafe_allow_html=True,
)


if forecast_clicked:

    with st.spinner(
        "Building the sales forecast..."
    ):

        try:

            forecast = (
                get_next_month_forecast.invoke({})
            )

            st.metric(
                label=(
                    f"Predicted Revenue — "
                    f"{forecast['forecast_month']}"
                ),
                value=(
                    f"£"
                    f"{forecast['predicted_revenue']:,.2f}"
                ),
            )

            st.caption(
                f"Random Forest · "
                f"{forecast['forecasted_days']} days · "
                f"MAE £{forecast['mae']:,.2f} · "
                f"RMSE £{forecast['rmse']:,.2f} · "
                f"R² {forecast['r2']:.2f}"
            )

        except Exception as e:

            st.error(
                f"Forecast unavailable: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
"""
<div class="footer">
MASHA · AI Business Data Analyst
· LangChain · Groq · RAG · Random Forest
</div>
""",
unsafe_allow_html=True,
)