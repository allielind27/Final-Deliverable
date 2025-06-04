# --- Streamlit Setup ---
import streamlit as st
st.set_page_config(page_title="Starbucks Forecasting", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import warnings
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# --- Load CSV ---
df = pd.read_csv("starbucks_financials_expanded.csv")
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('Q')

# --- Scrape CPI from FRED ---
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
        return float(cpi_elem.text.strip())
    except Exception as e:
        st.error(f"❌ Failed to scrape CPI: {e}")
        return None

if 'CPI' not in df.columns:
    df['CPI'] = float('nan')

latest_cpi = fetch_latest_cpi_scraper()
cpi_to_use = latest_cpi if latest_cpi else 320.321
df['CPI'].iloc[-4:] = cpi_to_use
st.markdown(f"**CPI used for forecast:** {cpi_to_use}")

# --- Clean Inputs for Model ---
revenue = df['revenue']
exog = df[['CPI', 'store_count']]

train_revenue = revenue[:-4]
test_revenue = revenue[-4:]
train_exog = exog[:-4]
test_exog = exog[-4:]

# Coerce and align
train_exog = train_exog.apply(pd.to_numeric, errors='coerce').dropna()
train_revenue = train_revenue.loc[train_exog.index]
test_exog = test_exog.apply(pd.to_numeric, errors='coerce')

# --- Fit Model ---
model = SARIMAX(train_revenue, exog=train_exog, order=(1,1,1), seasonal_order=(1,1,1,4))
results = model.fit(disp=False)
forecast = results.get_forecast(steps=4, exog=test_exog)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

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
st.line_chart(df[selected_vars])

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

# --- Summary ---
st.subheader("AI Summary")
st.markdown("""
Starbucks' revenue is forecasted to remain stable. However, forecasted revenue per store may exceed historical norms, 
indicating potential risk of overstatement. Average ticket size trends and external sentiment are important indicators 
to monitor alongside CPI-driven forecasting.
""")
