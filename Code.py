import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from datetime import datetime 
import warnings 
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# Custom title with smaller font size
st.markdown("""
    <h1 style='font-size: 24px;'>Starbucks Revenue Forecasting</h1>
""", unsafe_allow_html=True)

# --- Load Data ---
df = pd.read_csv("starbucks_financials_expanded.csv") 
df['date'] = pd.to_datetime(df['date']) 
df.set_index('date', inplace=True) 
df = df.asfreq('Q')

# --- CPI Handling via Web Scraping from FRED (Latest Value Only) ---
st.markdown("**Fetching Latest CPI Data from FRED Website**")
try:
    # Fetch the FRED CPIAUCSL page
    url = "https://fred.stlouisfed.org/series/CPIAUCSL"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # Parse the HTML to find the latest CPI value
    soup = BeautifulSoup(response.content, 'html.parser')
    # Search for the latest observation text and extract the adjacent value
    latest_obs = soup.find(string=lambda text: text and ("Apr 2025" in text or "Latest Observation" in text))
    if not latest_obs:
        raise ValueError("Could not find latest observation text on FRED page")
    
    # Find the parent element and look for the numeric value
    parent = latest_obs.find_parent()
    latest_cpi_elem = parent.find_next('span', class_=lambda x: x and any(kw in x.lower() for kw in ['value', 'data']))
    if not latest_cpi_elem or not latest_cpi_elem.text.strip().replace('.', '').isdigit():
        raise ValueError("Could not find valid CPI value near latest observation")
    
    latest_cpi = float(latest_cpi_elem.text.strip())
    st.write(f"Latest CPI Value: {latest_cpi}")
    
    # Assign the latest CPI value to the last 4 quarters in df['CPI']
    df['CPI'].iloc[-4:] = latest_cpi
    st.success("✅ Latest CPI data fetched successfully from FRED website.") 
except Exception as e:
    st.error(f"⚠️ Failed to fetch latest CPI data from FRED website: {e}")
    st.warning("Using fallback CPI value (320.321) to continue. Update data source for accurate forecasts.")
    df['CPI'].iloc[-4:] = 320.321  # Fallback value based on screenshot
    
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
