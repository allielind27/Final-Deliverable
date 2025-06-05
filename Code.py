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
import openai

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

# --- Data Loading ---
df = pd.read_csv("starbucks_financials_expanded.csv")
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('Q').fillna(method='ffill').fillna(method='bfill')  # Resample and fill NaNs

dunkin_df = pd.read_csv("dunkin_financials_generated.csv")
dunkin_df.columns = dunkin_df.columns.str.strip()
dunkin_df['date'] = pd.to_datetime(dunkin_df['date'])
dunkin_df.set_index('date', inplace=True)
dunkin_df = dunkin_df.asfreq('Q').fillna(method='ffill').fillna(method='bfill')  # Resample and fill NaNs

# Load Bruegger's data
brueggers_df = pd.read_csv("brueggers_financials_generated.csv")
brueggers_df.columns = brueggers_df.columns.str.strip()
brueggers_df['date'] = pd.to_datetime(brueggers_df['date'])
brueggers_df.set_index('date', inplace=True)
brueggers_df = brueggers_df.asfreq('Q').fillna(method='ffill').fillna(method='bfill')  # Resample and fill NaNs

# --- CPI Data Scraping ---
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

# --- CPI Integration ---
latest_cpi = fetch_latest_cpi_scraper()
cpi_to_use = latest_cpi if latest_cpi else 0
if 'CPI' not in df.columns or df['CPI'].isna().all():
    df['CPI'] = cpi_to_use
else:
    df['CPI'].fillna(cpi_to_use, inplace=True)

st.markdown("""
---
#### 📊 CPI Data Source
The Consumer Price Index (CPI) data used is sourced directly from the Federal Reserve Economic Data (FRED). 
This economic indicator serves as an exogenous input in the ARIMAX forecast to model the inflation impact on Starbucks’ revenue patterns.

**Series ID:** [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL)  
**Title:** Consumer Price Index for All Urban Consumers: All Items (Not Seasonally Adjusted)  
**Source:** U.S. Bureau of Labor Statistics  
""")
st.markdown(f"**CPI used for forecast:** {cpi_to_use} (fetched 2025-06-04 22:23)")

# --- User Input for Loyalty Members ---
st.markdown("""
---
### 👤 Adjust Loyalty Members Forecast
Before you begin reading the analysis, input the expected number of loyalty members for the upcoming quarter. This way, you can test different outcomes for future revenue based on your membership expectations.
""")

col1, col2 = st.columns([1, 3.4])
with col1:
    st.markdown(
        "<div style='padding-top: 34px; font-weight: bold;'>Expected loyalty members:</div>",
        unsafe_allow_html=True
    )
with col2:
    user_loyalty_members = st.number_input(
        label="",
        value=int(df['loyalty_members'].iloc[-1]),
        min_value=0,
        step=1000
    )

# --- Data Preparation for Forecasting ---
revenue = df['revenue']
exog = df[['CPI', 'loyalty_members']]  # Changed from store_count to loyalty_members
train_revenue = revenue[:-4]
test_revenue = revenue[-4:]
train_exog = exog[:-4].copy()
test_exog = exog[-4:].copy()

# Update the last row of test_exog with user input for loyalty_members
test_exog.iloc[-1, test_exog.columns.get_loc('loyalty_members')] = user_loyalty_members

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
    forecast = results.get_forecast(steps=4, exog=test_exog)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    forecast_mean.index = test_exog.index
    forecast_ci.index = test_exog.index
else:
    st.error("❌ Not enough clean training data to run the model. Please check your CPI/loyalty_members history.")
    st.stop()

st.markdown("---")

# --- Forecast Visualization ---
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
""")

# Align on shared dates across all three datasets
common_dates = df.index.intersection(dunkin_df.index).intersection(brueggers_df.index)

# Create two columns for side-by-side plots
col1, col2 = st.columns(2)

# --- Plot Average Ticket Size ---
with col1:
    st.subheader("Average Ticket Size")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    starbucks_avg = df.loc[common_dates, 'avg_ticket']
    dunkin_avg = dunkin_df.loc[common_dates, 'avg_ticket']
    brueggers_avg = brueggers_df.loc[common_dates, 'avg_ticket']
    ax1.plot(common_dates, starbucks_avg, label="Starbucks", color="#006241", linewidth=2)
    ax1.plot(common_dates, dunkin_avg, label="Dunkin", color="#FF6F00", linewidth=2)
    ax1.plot(common_dates, brueggers_avg, label="Bruegger's", color="#8B4513", linewidth=2)
    ax1.set_ylabel("Avg Ticket ($)")
    ax1.set_title("Average Ticket Size Over Time")
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
    ax2.plot(common_dates, brueggers_rev, label="Bruegger's", color="#8B4513", linewidth=2)
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
This section shows the overall percentage differences in Average Ticket Price and Revenue for Starbucks, Dunkin', and Bruegger's across the entire time period.
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
    companies = ["Starbucks", "Dunkin", "Bruegger's"]
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

st.markdown("""
---
### 🗞️ Sentiment Analysis
""")

# Sentiment Analysis
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

st.markdown("""
<hr>
<h2 style='text-align: center; margin-top: 20px;'>🤖 AI-Generated Summary</h2>
""", unsafe_allow_html=True)
