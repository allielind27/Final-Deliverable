# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import warnings
import requests
from bs4 import BeautifulSoup

# Configures the page
st.set_page_config(
    page_title="Starbucks Audit App", page_icon="☕", layout="wide"
)

# Title and summary of the app
st.markdown("""
    <h1 style='text-align: center;'>☕ Starbucks Revenue Forecasting App</h1>
    <h3 style='text-align: center;'>Powered by ARIMAX Modeling, Live Data, and Sentiment Analysis</h3>
    <hr>
    <div style='font-size: 16px; padding-top: 10px; text-align: left;'>
        <strong>📘 App Summary</strong><br><br>
        This app is a tool meant to aid audit teams with assessing the risk of revenue overstatement at Starbucks. 
    </div>
""", unsafe_allow_html=True)

warnings.filterwarnings("ignore")

# --- Load CSV ---
df = pd.read_csv("starbucks_financials_expanded.csv")
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('Q')

# --- Scrape CPI from FRED ---
@st.cache_data(ttl=3600)
def fetch_latest_cpi_scraper():
    try:
        url = "https://fred.stlouisfed.org/series/CPIAUCSL"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        cpi_elem = soup.find("span", class_="series-meta-observation-value")
        if not cpi_elem:
            raise ValueError("CPI value element not found.")
        cpi_value = cpi_elem.text.strip().replace(",", "")
        return float(cpi_value)
    except Exception as e:
        st.error(f"❌ Failed to scrape CPI: {e}")
        return None

# --- CPI Handling ---
latest_cpi = fetch_latest_cpi_scraper()
cpi_to_use = latest_cpi if latest_cpi else 0
if 'CPI' not in df.columns or df['CPI'].isna().all():
    df['CPI'] = cpi_to_use
else:
    df['CPI'].fillna(cpi_to_use, inplace=True)

st.markdown("""
---  
#### 📊 CPI Data Source

The Consumer Price Index (CPI) data used is sourced directly from the Federal Reserve Economic Data (FRED):

**Series ID:** [`CPIAUCSL`](https://fred.stlouisfed.org/series/CPIAUCSL)  
**Title:** Consumer Price Index for All Urban Consumers: All Items (Not Seasonally Adjusted)  
**Source:** U.S. Bureau of Labor Statistics 

This economic indicator serves as an exogenous input in the ARIMAX forecast to model the inflation impact on Starbucks’ revenue patterns.
""")

st.markdown(f"**CPI used for forecast:** {cpi_to_use}")

# --- Clean Inputs for Model ---
revenue = df['revenue']
exog = df[['CPI', 'store_count']]

# Split train and test
train_revenue = revenue[:-4]
test_revenue = revenue[-4:]

train_exog = exog[:-4].copy()
test_exog = exog[-4:].copy()

st.markdown("""
---  
### 🏪 Adjust Store Count Forecast""")

st.markdown("""
Before you begin reading the analysis, input the expected store count for the upcoming quarter. This way, you can test different outcomes for future revenue based on your location expectations.
""")

col1, col2 = st.columns([1, 3.4])

with col1:
    st.markdown(
        "<div style='padding-top: 34px; font-weight: bold;'>Expected store count for next period:</div>",
        unsafe_allow_html=True
    )

with col2:
    user_store_count = st.number_input(
        label="",
        value=int(test_exog['store_count'].iloc[-1]),
        min_value=0,
        step=10
    )
    
# Drop rows with NaNs
valid_mask = train_exog.notnull().all(axis=1)
train_exog = train_exog[valid_mask]
train_revenue = train_revenue[valid_mask]

# Final alignment
train_revenue, train_exog = train_revenue.align(train_exog, join='inner', axis=0)

# --- Fit Model ---
if train_revenue.shape[0] >= 12:
    model = SARIMAX(train_revenue, exog=train_exog, order=(1,1,1), seasonal_order=(1,1,1,4))
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=4, exog=test_exog)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    forecast_mean.index = test_exog.index
    forecast_ci.index = test_exog.index
else:
    st.error("❌ Not enough clean training data to run the model. Please check your CPI/store_count history.")
    st.stop()

st.markdown("""
<h2 style='text-align: center; margin-top: 40px;'>📈 Revenue Forecast Model</h2>
""", unsafe_allow_html=True)

# --- Forecast Plot ---
st.title("Forecast vs Actual")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(revenue.index, revenue, label='Actual Revenue', color='blue')
ax.plot(forecast_mean.index, forecast_mean, label='Forecasted Revenue', color='orange')
ax.fill_between(forecast_mean.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='orange', alpha=0.3)
ax.set_title("Revenue Forecast")
ax.set_ylabel("Revenue (in millions)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Risk Flag ---
if risk_flag:
    st.error("⚠️ Risk: Forecasted revenue per store is unusually high.")
else:
    st.success("✅ Revenue per store forecast is reasonable.")

# --- Revenue per Store Check ---
latest_store_count = df['store_count'].iloc[-4:]
rev_per_store_forecast = forecast_mean / latest_store_count.values
historical_ratio = (train_revenue / train_exog['store_count']).mean()
risk_flag = any(rev_per_store_forecast > 1.25 * historical_ratio)

# --- Average Ticket Insight ---
st.subheader("Average Ticket Size Insight")
avg_ticket_recent = df['avg_ticket'].iloc[-4:]
avg_ticket_mean = df['avg_ticket'].mean()
st.line_chart(df['avg_ticket'], use_container_width=True)

if avg_ticket_recent.mean() > 1.1 * avg_ticket_mean:
    st.warning("⚠️ Average ticket size is significantly above average.")
elif avg_ticket_recent.mean() < 0.9 * avg_ticket_mean:
    st.info("ℹ️ Average ticket size is below long-term average.")
else:
    st.success("✅ Ticket size is stable.")

# --- Sentiment Analysis ---
st.subheader("Earnings Headline Sentiment")
headlines = [
    "Starbucks beats expectations with strong Q1 sales",
    "Concerns arise over Starbucks' China performance",
    "Starbucks forecasts modest growth despite inflation"
]
positive_keywords = ["beats", "strong", "growth", "record", "positive"]
negative_keywords = ["concerns", "misses", "slowdown", "decline", "drop"]

def score_sentiment(text):
    text = text.lower()
    return sum(word in text for word in positive_keywords) - sum(word in text for word in negative_keywords)

sentiments = [score_sentiment(h) for h in headlines]
for h, s in zip(headlines, sentiments):
    sentiment_type = "🟢 Positive" if s > 0 else "🔴 Negative" if s < 0 else "🟡 Neutral"
    st.write(f"{sentiment_type}: {h}")

# --- Benchmarking ---
st.subheader("Industry Peer Comparison")
peer_data = pd.DataFrame({
    'Company': ['Starbucks', 'Dunkin', 'Dutch Bros'],
    'Revenue Growth (%)': [12.0, 9.5, 15.2],
    'Avg Ticket ($)': [6.15, 5.80, 6.45]
})
st.dataframe(peer_data)

# --- Interactive Plot ---
st.subheader("Explore Starbucks KPIs")
selected_vars = st.multiselect("Select variables to plot:", df.columns, default=['revenue', 'store_count'])
if selected_vars:
    st.line_chart(df[selected_vars])
