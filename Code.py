import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import warnings
import matplotlib.dates
from openai import OpenAI
import plotly.express as px

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="Starbucks Audit App",
    page_icon="☕",
    layout="wide"
)

# --- App Header ---
st.markdown("""
    <h1 style='text-align: center;'>☕ Starbucks Revenue Forecasting App</h1>
    <h3 style='text-align: center;'>Powered by ARIMAX Modeling, Live Data, and Sentiment Analysis</h3>
    <hr>
""", unsafe_allow_html=True)

# --- App Summary ---
st.markdown("""
### 📘 App Thesis
Starbucks’ revenue appears to be overstated because CPI, Loyalty Membership, and Average Ticket Price are not related with revenue growth.
This application is meant to provide automatic analysis to determine the risk of overstated revenue.
""")

# --- CPI Data Scraping (Historical from FRED) ---
@st.cache_data(ttl=3600)
def fetch_historical_cpi(dates):
    try:
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        api_key = "5140f3a1760f911c23923852a41c82d3"  
        params = {
            "series_id": "CPIAUCSL",
            "api_key": api_key,
            "file_type": "json",
            "observation_start": dates.min().strftime('%Y-%m-%d'),
            "observation_end": dates.max().strftime('%Y-%m-%d')
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()["observations"]
        cpi_data = {pd.to_datetime(d["date"]): float(d["value"]) for d in data if d["value"] != "."}
        return pd.Series(cpi_data).reindex(dates, method='ffill').fillna(method='bfill')
    except Exception as e:
        st.error(f"❌ Failed to fetch historical CPI: {e}")
        return pd.Series(index=dates, data=0.0)

# --- CPI Integration ---
df['CPI'] = fetch_historical_cpi(df.index)

st.markdown("""
---
#### 📊 CPI Data Source
The Consumer Price Index (CPI) data used is sourced directly from the Federal Reserve Economic Data (FRED). 
This economic indicator serves as an exogenous input in the ARIMAX forecast to model the inflation impact on Starbucks’ revenue patterns.

**Series ID:** [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL)  
**Title:** Consumer Price Index for All Urban Consumers: All Items (Not Seasonally Adjusted)  
**Source:** U.S. Bureau of Labor Statistics  
""")

# --- User Input for CPI ---
st.markdown("""
---
### 👤 Adjust CPI Forecast
Input the expected CPI value for the upcoming quarter to test different inflation scenarios for Starbucks' revenue forecast.
""")

col1, col2 = st.columns([1, 3.4])
with col1:
    st.markdown(
        "<div style='padding-top: 34px; font-weight: bold;'>Expected CPI:</div>",
        unsafe_allow_html=True
    )
with col2:
    user_cpi = st.number_input(
        label="",
        value=float(df['CPI'].iloc[-1]) if not df['CPI'].iloc[-1] == 0 else 300.0,  # Default to last CPI or 300 if zero
        min_value=0.0,
        step=0.1
    )

# --- Data Preparation for Forecasting ---
revenue = df['revenue']
exog = df[['CPI']]  # Removed loyalty_members, using only CPI
train_revenue = revenue[:-1]  # Train on all but the last quarter
test_revenue = revenue[-1:]  # Test on the last quarter
train_exog = exog[:-1].copy()
test_exog = exog[-1:].copy()

# Update the last row of test_exog with user input for CPI
test_exog.iloc[-1, test_exog.columns.get_loc('CPI')] = user_cpi

# Clean data: remove rows with NaNs and align
valid_mask = train_exog.notnull().all(axis=1)
train_exog = train_exog[valid_mask]
train_revenue = train_revenue[valid_mask]
train_revenue, train_exog = train_revenue.align(train_exog, join='inner', axis=0)

# --- SARIMAX Model and Forecasting ---
st.markdown("""
<hr>
<h2 style='text-align: center; margin-top: 20px;'>📈 Revenue Forecast Model</h2>
""", unsafe_allow_html=True)

if train_revenue.shape[0] >= 12:
    model = SARIMAX(train_revenue, exog=train_exog, order=(1,1,1), seasonal_order=(1,1,1,4))
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=1, exog=test_exog)  # Forecast only next quarter
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    forecast_mean.index = test_exog.index
    forecast_ci.index = test_exog.index
else:
    st.error("❌ Not enough clean training data to run the model. Please check your CPI history.")
    st.stop()

st.markdown("---")

# --- Forecast Visualization ---
st.title("Forecast vs Actual")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(revenue.index, revenue, label='Actual Revenue', color='blue')
ax.plot(forecast_mean.index, forecast_mean, label='Forecasted Revenue', color='orange')
ax.fill_between(forecast_mean.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='orange', alpha=0.3)
ax.set_title("Starbucks Revenue Forecast (Next Quarter)")
ax.set_ylabel("Revenue (in millions)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Forecast Results Summary Table ---
st.markdown("""
---
### 📊 Forecast Results
The table below compares forecasted revenue to actual revenue for the next four quarters, based on the model using CPI and loyalty membership data.
""")

# Prepare data
quarters = forecast_mean.index.strftime('%Y-%m')
forecasted = forecast_mean.round(2)
actual = test_revenue.reindex(forecast_mean.index).round(2)
pct_diff = ((forecasted - actual) / actual * 100).round(2)

# Combine into one DataFrame
results_df = pd.DataFrame({
    'Date': quarters,
    'Forecasted Revenue ($M)': forecasted.values,
    'Actual Revenue ($M)': actual.values,
    '% Difference': pct_diff.values
})

# Display the DataFrame
st.dataframe(results_df)

# Build color map for the bar chart
colors = ['red' if abs(val) > 5 else 'gray' for val in pct_diff]

# Plot bar chart for percentage difference
st.markdown("""
---
### 📉 % Difference Between Forecasted and Actual Revenue
""")

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(quarters, pct_diff, color=colors)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('% Difference')
ax.set_title('Quarter')
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

st.pyplot(fig)

# Show alert if any % difference exceeds ±5%
if any(abs(pct_diff) > 5):
    st.warning("⚠️ One or more forecasted revenues differ from actuals by more than 5%. This may indicate a risk of revenue overstatement or model inaccuracy related to loyalty membership or CPI assumptions.")
else:
    st.success("✅ Forecasted revenue is within 5% of actuals across all quarters.")

st.markdown("""
<hr>
<h2 style='text-align: center; margin-top: 20px;'>🔍 Additional Insights</h2>
""", unsafe_allow_html=True)

# --- KPI Insights ---
st.markdown("""
---
### 📊 KPI Insights
This section provides visualizations of Average Ticket Price, and Revenue over time for Starbucks and it's main competitors.
""")

# Align on shared dates across all three datasets
common_dates = df.index.intersection(dunkin_df.index).intersection(brueggers_df.index)

# Create two columns for side-by-side plots
col1, col2 = st.columns(2)

# --- Plot Average Ticket Size ---
with col1:
    st.subheader("Average Ticket Price")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    starbucks_avg = df.loc[common_dates, 'avg_ticket']
    dunkin_avg = dunkin_df.loc[common_dates, 'avg_ticket']
    brueggers_avg = brueggers_df.loc[common_dates, 'avg_ticket']
    ax1.plot(common_dates, starbucks_avg, label="Starbucks", color="#006241", linewidth=2)
    ax1.plot(common_dates, dunkin_avg, label="Dunkin", color="#FF6F00", linewidth=2)
    ax1.plot(common_dates, brueggers_avg, label="Dutch Bro's", color="#8B4513", linewidth=2)
    ax1.set_ylabel("Avg Ticket ($)")
    ax1.set_title("Average Ticket Price Over Time")
    ax1.legend()
    ax1.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

# --- Plot Revenue ---
with col2:
    st.subheader("Revenue Over Time")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    starbucks_rev = df.loc[common_dates, 'revenue']
    dunkin_rev = dunkin_df.loc[common_dates, 'revenue']
    brueggers_rev = brueggers_df.loc[common_dates, 'revenue']
    ax2.plot(common_dates, starbucks_rev, label="Starbucks", color="#006241", linewidth=2)
    ax2.plot(common_dates, dunkin_rev, label="Dunkin", color="#FF6F00", linewidth=2)
    ax2.plot(common_dates, brueggers_rev, label="Dutch Bro's", color="#8B4513", linewidth=2)
    ax2.set_ylabel("Revenue ($M)")
    ax2.set_title("Revenue Over Time")
    ax2.legend()
    ax2.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# --- Overall Percentage Differences in KPIs ---
st.markdown("""
---
### 📊 Overall Percentage Differences in KPIs
This section shows the overall percentage differences in Average Ticket Price and Revenue for Starbucks, Dunkin', and Dutch Bro's across the entire time period.
""")

# Calculate overall percentage difference for each metric
def calculate_overall_pct_diff(series):
    first_value = series.iloc[0]
    last_value = series.iloc[-1]
    return ((last_value - first_value) / first_value) * 100  # Overall % change

# Compute overall percentage differences for each company
starbucks_avg_pct = calculate_overall_pct_diff(df.loc[common_dates, 'avg_ticket'])
dunkin_avg_pct = calculate_overall_pct_diff(dunkin_df.loc[common_dates, 'avg_ticket'])
brueggers_avg_pct = calculate_overall_pct_diff(brueggers_df.loc[common_dates, 'avg_ticket'])

starbucks_rev_pct = calculate_overall_pct_diff(df.loc[common_dates, 'revenue'])
dunkin_rev_pct = calculate_overall_pct_diff(dunkin_df.loc[common_dates, 'revenue'])
brueggers_rev_pct = calculate_overall_pct_diff(brueggers_df.loc[common_dates, 'revenue'])

# Create two columns for side-by-side plots
col1, col2 = st.columns(2)

# --- Bar Graph for Overall % Difference in Average Ticket Price ---
with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    
    # Data for plotting
    companies = ["Starbucks", "Dunkin", "Dutch Bro's"]
    avg_pcts = [starbucks_avg_pct, dunkin_avg_pct, brueggers_avg_pct]
    colors = ["#006241", "#FF6F00", "#8B4513"]
    
    # Plot bars
    bars = ax1.bar(companies, avg_pcts, color=colors, width=0.6)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        label = f"{height:.2f}%"
        if height >= 0:
            ax1.text(bar.get_x() + bar.get_width()/2, height, label, ha='center', va='bottom', fontsize=10)
        else:
            ax1.text(bar.get_x() + bar.get_width()/2, height, label, ha='center', va='top', fontsize=10)
    
    # Customize plot
    ax1.set_ylabel("% Change in Avg Ticket")
    ax1.set_title("Overall % Change in Avg Ticket Price")
    ax1.grid(True, axis='y')
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

# --- Bar Graph for Overall % Difference in Revenue ---
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    
    # Data for plotting
    rev_pcts = [starbucks_rev_pct, dunkin_rev_pct, brueggers_rev_pct]
    
    # Plot bars
    bars = ax2.bar(companies, rev_pcts, color=colors, width=0.6)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        label = f"{height:.2f}%"
        if height >= 0:
            ax2.text(bar.get_x() + bar.get_width()/2, height, label, ha='center', va='bottom', fontsize=10)
        else:
            ax2.text(bar.get_x() + bar.get_width()/2, height, label, ha='center', va='top', fontsize=10)
    
    # Customize plot
    ax2.set_ylabel("% Change in Revenue")
    ax2.set_title("Overall % Change in Revenue")
    ax2.grid(True, axis='y')
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# --- Risk Pop-Up for Diverging Revenue and Ticket Price ---
# Check if revenue and avg ticket price are moving in opposite directions
risk_companies = []
for company, rev_pct, avg_pct in zip(companies, rev_pcts, avg_pcts):
    # Opposite directions: one is positive, the other is negative
    if (rev_pct > 0 and avg_pct < 0) or (rev_pct < 0 and avg_pct > 0):
        risk_companies.append(company)

# Display warning if any companies are at risk
if risk_companies:
    st.warning(
        f"⚠️ **Risk Alert**: Revenue and Average Ticket Price are moving in opposite directions for {', '.join(risk_companies)}. "
        "This may indicate potential pricing or demand inconsistencies affecting revenue trends."
    )

st.markdown("""
---
### 🗞️ Sentiment Analysis
""")

# Initialize session state for headlines
if 'headlines' not in st.session_state:
    st.session_state.headlines = [
        "Starbucks beats expectations with strong Q1 sales",
        "Starbucks reports weak earnings",
        "Starbucks releases new drink"
    ]

# Keyword lists
positive_keywords = [
    "beats expectations", "exceeds expectations", "exceeds forecasts", "exceeds guidance", "strong",
    "strength", "growth", "record", "positive", "profit", "increased", "expansion", "surged", "surge",
    "resilient", "improved", "robust", "momentum", "uptrend", "tops", "outperforms", "success",
    "accelerated", "recovery", "rebound", "bullish", "gained", "sustainable", "upside", "profitable",
    "stable", "expanded", "improves", "boost", "advances", "resurgence", "rebounded", "excels",
    "cashflow", "resilience", "uptick", "elevated", "surpassing", "superior", "scaling", "gaining",
    "thriving", "leadership", "stability", "peak", "streamlined", "enhanced", "consistent",
    "reaffirmed", "beneficial", "rising", "maximized", "favorably", "delivered", "dividend", "upgrade",
    "oversubscribed", "greenlighted", "licensed", "compliant", "settled", "cleared", "dismissed",
    "acquitted", "authorized", "reinstated", "accretive", "liquidity", "refinanced", "renewed",
    "restructured", "deleveraged", "covered", "hedged", "ratified", "voted", "approved", "unanimous",
    "surplus", "milestone", "capitalized", "strengthened", "fortified", "certified", "empowered",
    "revitalized", "endorsed", "appointed", "nominated", "balanced", "merged", "diversified",
    "aligned", "synergistic", "acquired", "launched", "adopted", "standardized", "cohesive",
    "integrated", "optimized", "raises guidance", "raises outlook", "record sales", "market leader",
    "breakthrough", "innovation", "patent granted", "expansion plans", "new markets", "efficient",
    "evolving", "operational excellence", "inventory efficiency", "cost optimization",
    "supply chain recovery", "traffic gains", "demand resilience", "amazing", "awesome", "brilliant",
    "cool", "delightful", "excellent", "fantastic", "friendly", "fun", "great", "happy", "helpful",
    "inspiring", "joyful", "kind", "loved", "lovely", "motivated", "nice", "outstanding", "pleasant",
    "satisfying", "smart", "smooth", "stellar", "stronger", "super", "terrific", "thankful",
    "trusted", "welcoming", "wonderful", "admirable", "beautiful", "cheerful", "commendable",
    "courteous", "dedicated", "dependable", "encouraging", "energetic", "enthusiastic", "fair",
    "favorable", "funny", "genuine", "grateful", "honest", "intelligent", "loving", "neat",
    "nice-looking", "peaceful", "polite", "positive-minded", "quick", "refreshing", "respectful",
    "safe", "sharp", "skillful", "supportive", "tidy", "upbeat", "vibrant", "warm", "wise", "worthy"
]
negative_keywords = [
    "misses expectations", "below expectations", "earnings miss", "revenue miss", "shortfall", "decline",
    "drop", "slump", "loss", "cut", "downgrade", "underperforms", "underperformed", "disappointing",
    "weak", "volatility", "downtrend", "slowdown", "plummet", "collapse", "unexpected", "shrinking",
    "softness", "suffers", "reduced", "cutting", "glut", "headwinds", "deficit", "attrition",
    "cautious", "delay", "weaker", "slower", "constrained", "challenging", "stagnant", "squeezed",
    "pullback", "impacted", "shortage", "ineffective", "unfavorable", "overexposed", "problematic",
    "unmet", "unresolved", "noncompliant", "penalized", "fined", "recalled", "delisted", "downgraded",
    "warned", "delayed", "withdrawn", "cancelled", "diluted", "sued", "suspension", "restated",
    "investigated", "charged", "violated", "exposed", "bankrupt", "insolvent", "declined", "fraud",
    "default", "divestment", "writeoff", "abandoned", "resigned", "terminated", "fired", "lawsuit",
    "scrapped", "risked", "underfunded", "worsened", "triggered", "noncompliance", "infringed",
    "litigated", "flagged", "breach", "blacklisted", "subpoenaed", "dismissed", "weakness",
    "instability", "misstated", "misclassified", "refuted", "pressured", "strained", "overstated",
    "disqualified", "malfunction", "revoked", "restatement", "uncertain", "risky", "speculative",
    "margin squeeze", "revenue decline", "profit warning", "cost overrun", "budget overrun",
    "inefficiency", "supply chain issues", "staff shortage", "store closures", "inventory glut",
    "slower conversion", "operational challenges", "system failure", "legal battles", "lawsuit",
    "litigation", "struggling", "facing charges", "regulatory issues", "annoying", "awful", "bad",
    "boring", "broken", "careless", "cold", "confusing", "cruel", "damaged", "dirty", "disappointing",
    "dull", "frustrating", "gloomy", "gross", "hard", "horrible", "hostile", "hurtful", "ignorant",
    "impolite", "inaccurate", "inconsiderate", "inept", "lazy", "loud", "mean", "messy", "nasty",
    "negative", "noisy", "painful", "poor", "rude", "sad", "scary", "selfish", "shameful", "shocking",
    "slow", "smelly", "stale", "stressful", "stupid", "tense", "terrible", "thoughtless", "toxic",
    "ugly", "unbearable", "unclear", "unfair", "unfriendly", "unhappy", "unpleasant", "unreliable",
    "upset", "useless", "vague", "worthless", "wrong"
]
negation_words = [
    "not", "no", "never", "none", "without", "rarely", "hardly", "barely", "didn't", "doesn't",
    "wasn't", "isn't", "aren't", "can't", "couldn't", "won't", "hasn't", "haven't", "shouldn't",
    "wouldn't", "neither", "nor", "fails to", "fail to", "lacks", "unmet", "avoids", "excludes",
    "incomplete", "short of", "absence of", "devoid of", "ain’t", "refuses", "stops", "prevents"
]

# Sentiment scoring function with exact phrase matching and single-word fallback
def score_sentiment(text):
    text = text.lower()
    score = 0
    matched_phrases = set()
    words = text.split()
    
    # Sort keywords by length (longest first) to prioritize multi-word phrases
    sorted_positive_keywords = sorted(positive_keywords, key=len, reverse=True)
    sorted_negative_keywords = sorted(negative_keywords, key=len, reverse=True)
    
    # Check for positive phrases
    for phrase in sorted_positive_keywords:
        phrase_words = phrase.split()
        for i in range(len(words) - len(phrase_words) + 1):
            if ' '.join(words[i:i + len(phrase_words)]) == phrase:
                preceding_text = ' '.join(words[:i])
                if not any(n in preceding_text for n in negation_words):
                    score += 1
                else:
                    score -= 1
                matched_phrases.add(phrase)
                break
    
    # Check for negative phrases
    for phrase in sorted_negative_keywords:
        phrase_words = phrase.split()
        for i in range(len(words) - len(phrase_words) + 1):
            if ' '.join(words[i:i + len(phrase_words)]) == phrase:
                preceding_text = ' '.join(words[:i])
                if not any(n in preceding_text for n in negation_words):
                    score -= 1
                else:
                    score += 1
                matched_phrases.add(phrase)
                break
    
    # Fallback: Check single words in keywords (if no phrases matched)
    if score == 0:  # Only check single words if no phrases matched
        for word in words:
            if word in [kw for kw in positive_keywords if ' ' not in kw] and word not in matched_phrases:
                preceding_text = ' '.join(words[:words.index(word)])
                if not any(n in preceding_text for n in negation_words):
                    score += 1
                else:
                    score -= 1
                matched_phrases.add(word)
            elif word in [kw for kw in negative_keywords if ' ' not in kw] and word not in matched_phrases:
                preceding_text = ' '.join(words[:words.index(word)])
                if not any(n in preceding_text for n in negation_words):
                    score -= 1
                else:
                    score += 1
                matched_phrases.add(word)
    
    return score

# Create two columns for side-by-side input
col1, col2 = st.columns(2)

# Add headline section
with col1:
    st.write("**Add a new headline**")
    new_headline = st.text_input("Enter a headline:", key="add_headline")
    if st.button("Add Headline"):
        if new_headline.strip():
            st.session_state.headlines.append(new_headline)
            st.success(f"Added: {new_headline}")
        else:
            st.warning("Please enter a valid headline.")

# Remove headline section
with col2:
    st.write("**Remove a headline**")
    headline_to_remove = st.selectbox("Select a headline to remove:", [""] + st.session_state.headlines, key="remove_headline", index=0)
    if st.button("Remove Headline"):
        if headline_to_remove:
            st.session_state.headlines.remove(headline_to_remove)
            st.success(f"Removed: {headline_to_remove}")
        else:
            st.warning("Please select a headline to remove.")

# Display sentiment results
st.write("**Sentiment Results**")
if st.session_state.headlines:
    sentiments = [score_sentiment(h) for h in st.session_state.headlines]
    for h, s in zip(st.session_state.headlines, sentiments):
        sentiment_type = "🟢 Positive" if s > 0 else "🔴 Negative" if s < 0 else "🟡 Neutral"
        st.write(f"{sentiment_type} (Score: {s}): {h}")
    
    # Visualize sentiment distribution
    sentiment_counts = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Count": [
            len([s for s in sentiments if s > 0]),
            len([s for s in sentiments if s < 0]),
            len([s for s in sentiments if s == 0])
        ]
    })
    fig = px.bar(sentiment_counts, x="Sentiment", y="Count", title="Sentiment Analysis of Earnings Headlines",
                 color="Sentiment", color_discrete_map={"Positive": "#00cc00", "Negative": "#ff3333", "Neutral": "#ffcc00"})
    st.plotly_chart(fig)
else:
    st.write("No headlines to analyze.")

# Calculate summary inputs dynamically
# Forecast accuracy (% difference)
pct_diff = ((forecasted - actual) / actual * 100).round(2)
max_forecast_deviation = abs(pct_diff).max()

# Average ticket price trends
starbucks_avg_pct = calculate_overall_pct_diff(df.loc[common_dates, 'avg_ticket'])
dunkin_avg_pct = calculate_overall_pct_diff(dunkin_df.loc[common_dates, 'avg_ticket'])
brueggers_avg_pct = calculate_overall_pct_diff(brueggers_df.loc[common_dates, 'avg_ticket'])

# Sentiment analysis
sentiment_counts_dict = sentiment_counts.set_index("Sentiment")["Count"].to_dict()
total_headlines = sum(sentiment_counts_dict.values())
positive_pct = (sentiment_counts_dict.get("Positive", 0) / total_headlines * 100) if total_headlines > 0 else 0
negative_pct = (sentiment_counts_dict.get("Negative", 0) / total_headlines * 100) if total_headlines > 0 else 0
neutral_pct = (sentiment_counts_dict.get("Neutral", 0) / total_headlines * 100) if total_headlines > 0 else 0

# Dynamic summary prompt
summary_prompt = f"""
You are an AI financial assistant reviewing a quarterly report for Starbucks.

TASK:
Write a short audit-focused summary (under 100 words) evaluating whether revenue appears overstated, based on:
- Forecast accuracy: Forecasted revenues deviated by up to {max_forecast_deviation:.1f}% from actuals.
- Average transaction size trends: Starbucks' average ticket price changed by {starbucks_avg_pct:.1f}%, compared to Dunkin ({dunkin_avg_pct:.1f}%) and Dutch Bros ({brueggers_avg_pct:.1f}%).
- Public sentiment: Sentiment analysis shows {positive_pct:.0f}% Positive, {negative_pct:.0f}% Negative, {neutral_pct:.0f}% Neutral.
Use clear, professional language suitable for a boardroom setting.
"""

# Initialize OpenAI client with API key from secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a financial audit assistant."},
        {"role": "user", "content": summary_prompt}
    ],
    temperature=0.4
)

ai_summary = response.choices[0].message.content

# Display the AI-generated summary
st.markdown("""
---
### 📝 AI-Generated Audit Summary
""")
st.write(ai_summary)
