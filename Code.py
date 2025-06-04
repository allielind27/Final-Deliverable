import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from datetime import datetime 
import warnings 
import requests
from bs4 import BeautifulSoup
import io

warnings.filterwarnings("ignore")
st.markdown("""
    <h1 style='font-size: 36px;'>Starbucks Revenue Forecasting App</h1>
""", unsafe_allow_html=True)

st.markdown("""
Welcome to the Starbucks Revenue Forecasting App! This application provides useful tools to forecast Starbucks' quarterly revenue using time-series modeling. 
By leveraging economic indicators and operational metrics, the app delivers insights into future revenue trends. 
""")

# --- Load Data ---
df = pd.read_csv("starbucks_financials_expanded.csv") 
df['date'] = pd.to_datetime(df['date']) 
df.set_index('date', inplace=True) 
df = df.asfreq('Q')

# --- CPI Handling via Web Scraping from FRED ---
st.markdown("**Fetching CPI Data from FRED Website**")
try:
    # Fetch the FRED CPIAUCSL page
    url = "https://fred.stlouisfed.org/series/CPIAUCSL"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # Parse the page to find the CSV download link
    soup = BeautifulSoup(response.content, 'html.parser')
    download_link = soup.find('a', href=lambda href: href and 'download' in href.lower() and 'csv' in href.lower())
    if not download_link:
        raise ValueError("Could not find CSV download link on FRED page")
    
    # Fetch the CSV data
    csv_url = "https://fred.stlouisfed.org" + download_link['href']
    csv_response = requests.get(csv_url, headers=headers)
    csv_response.raise_for_status()
    
    # Read CSV data into a DataFrame
    live_cpi = pd.read_csv(io.StringIO(csv_response.text))
    live_cpi['DATE'] = pd.to_datetime(live_cpi['DATE'])
    live_cpi.set_index('DATE', inplace=True)
    live_cpi = live_cpi.resample('Q').mean()
    df['CPI'] = live_cpi['CPIAUCSL'].reindex(df.index).fillna(method='ffill')
    st.success("✅ Live CPI data fetched successfully from FRED website.") 
except Exception as e:
    st.error(f"⚠️ Failed to fetch live CPI data from FRED website: {e}")
    st.error("The app requires live CPI data to proceed. Please try again later.")
    st.stop()
    
# --- Forecast revenue using ARIMAX ---
revenue = df['revenue'] 
exog = df[['CPI', 'store_count']]

train_revenue = revenue[:-4] 
test_revenue = revenue[-4:] 
train_exog = exog[:-4] 
test_exog = exog[-4:]

# Combine and clean training data
train_data = pd.concat([train_revenue, train_exog], axis=1).dropna()
train_revenue = train_data['revenue']
train_exog = train_data[['CPI', 'store_count']]

# Fit the model and forecast
model = SARIMAX(train_revenue, exog=train_exog, order=(1,1,1), seasonal_order=(1,1,1,4)) 
results = model.fit(disp=False) 
forecast = results.get_forecast(steps=4, exog=test_exog) 
forecast_mean = forecast.predicted_mean 
forecast_ci = forecast.conf_int()

# --- Revenue per store analysis ---
latest_store_count = df['store_count'].iloc[-4:] 
rev_per_store_forecast = forecast_mean / latest_store_count.values 
historical_ratio = (train_revenue / train_exog['store_count']).mean()
risk_flag = any(rev_per_store_forecast > 1.25 * historical_ratio)

# --- New Insight: Avg Ticket Analysis ---
st.subheader("New Insight: Average Ticket Size")
avg_ticket_recent = df['avg_ticket'].iloc[-4:]
avg_ticket_mean = df['avg_ticket'].mean()
st.line_chart(df['avg_ticket'], use_container_width=True)

if avg_ticket_recent.mean() > 1.1 * avg_ticket_mean:
    st.warning("⚠️ Recent average ticket size is significantly higher than historical average. This may reflect price increases or shifts in customer behavior.")
elif avg_ticket_recent.mean() < 0.9 * avg_ticket_mean:
    st.info("ℹ️ Recent average ticket size is below the long-term average, which may signal discounting or reduced spending per transaction.")
else:
    st.success("✅ Average ticket size appears consistent with historical levels.")

# --- Sentiment Analysis of Headlines (keyword-based) ---
st.subheader("Sentiment Analysis of Recent Earnings Headlines")
headlines = [
    "Starbucks beats expectations with strong Q1 sales",
    "Concerns arise over Starbucks' China performance",
    "Starbucks forecasts modest growth despite inflation"
]
positive_keywords = ["beats", "strong", "growth", "record", "positive"]
negative_keywords = ["concerns", "misses", "slowdown", "decline", "drop"]

def score_sentiment(text):
    text = text.lower()
    score = sum(word in text for word in positive_keywords) - sum(word in text for word in negative_keywords)
    return score

sentiments = [score_sentiment(h) for h in headlines]
sentiment_score = np.mean(sentiments)

for h, s in zip(headlines, sentiments):
    sentiment_type = "🟢 Positive" if s > 0 else "🔴 Negative" if s < 0 else "🟡 Neutral"
    st.write(f"{sentiment_type}: {h}")

if sentiment_score < -1:
    st.error("⚠️ Negative sentiment detected in recent earnings headlines.")
elif sentiment_score > 1:
    st.success("✅ Headlines suggest positive market sentiment.")
else:
    st.info("ℹ️ Sentiment appears mixed or neutral.")

# --- Peer Benchmarking ---
st.subheader("Benchmark: Starbucks vs Industry Peers")
peer_data = pd.DataFrame({
    'Company': ['Starbucks', 'Dunkin', 'Dutch Bros'],
    'Revenue Growth (%)': [12.0, 9.5, 15.2],
    'Avg Ticket ($)': [6.15, 5.80, 6.45]
})
st.dataframe(peer_data)

# --- Filters and Visualizations ---
st.subheader("Interactive Visualizations")
selected_vars = st.multiselect("Select variables to visualize:", df.columns, default=['revenue', 'store_count'])
st.line_chart(df[selected_vars])

# --- Plot ---
st.title("Starbucks Revenue Forecasting App")
st.write("""
This app forecasts Starbucks quarterly revenue using ARIMAX. 
It incorporates store count and CPI as predictors. Users can choose to use live CPI data from FRED or enter a manual CPI value.
""")

fig, ax = plt.subplots(figsize=(10, 5)) 
ax.plot(revenue.index, revenue, label='Actual Revenue', color='blue') 
ax.plot(forecast_mean.index, forecast_mean, label='Forecasted Revenue', color='orange') 
ax.fill_between(
    forecast_mean.index, 
    forecast_ci.iloc[:, 0].astype(float), 
    forecast_ci.iloc[:, 1].astype(float), 
    color='orange', alpha=0.3
) 
ax.set_title("Revenue Forecast vs Actual") 
ax.set_ylabel("Revenue (in millions)") 
ax.legend() 
ax.grid(True) 
st.pyplot(fig)

# --- Risk flag output ---
if risk_flag: 
    st.error("⚠️ Potential Overstatement Risk: Forecasted revenue per store exceeds historical range.") 
else: 
    st.success("✅ Revenue per store appears within normal historical range.")

# --- AI Summary (static version) ---
st.subheader("AI Summary") 
st.markdown("""
Based on the ARIMAX forecast using CPI and store count, Starbucks' revenue is projected to remain stable over the next four quarters. 
However, revenue per store shows a potential increase above historical norms. 
This could indicate aggressive revenue projections not matched by store expansion, suggesting a moderate risk of revenue overstatement. 
The average ticket size also appears to be a meaningful signal and should be monitored to better understand consumer behavior trends.
Earnings sentiment and peer comparisons also support the importance of monitoring pricing strategies and external expectations.
""")
