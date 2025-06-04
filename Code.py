# --- Title ---
st.markdown("<h1 style='font-size: 24px;'>Starbucks Revenue Forecasting</h1>", unsafe_allow_html=True)

# --- Load Excel Data ---
df = pd.read_excel("starbucks_financials_expanded.xlsx") 
df.columns = df.columns.str.strip()  # Clean column names
df['date'] = pd.to_datetime(df['date']) 
df.set_index('date', inplace=True) 
df = df.asfreq('Q')

# --- Web Scraper for CPI ---
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
        st.error(f"❌ Web scraping CPI failed: {e}")
        return None

# --- Apply CPI to DataFrame ---
if 'CPI' not in df.columns:
    df['CPI'] = float('nan')

latest_cpi = fetch_latest_cpi_scraper()
cpi_to_use = latest_cpi if latest_cpi else 320.321
df['CPI'].iloc[-4:] = cpi_to_use
st.markdown(f"**Latest CPI Value Used:** {cpi_to_use}")

# --- Forecasting with ARIMAX ---
revenue = df['revenue'] 
exog = df[['CPI', 'store_count']]

train_revenue = revenue[:-4] 
test_revenue = revenue[-4:] 
train_exog = exog[:-4] 
test_exog = exog[-4:]

train_data = pd.concat([train_revenue, train_exog], axis=1).dropna()
train_revenue = train_data['revenue']
train_exog = train_data[['CPI', 'store_count']]

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

# --- Average Ticket Insight ---
st.subheader("New Insight: Average Ticket Size")
avg_ticket_recent = df['avg_ticket'].iloc[-4:]
avg_ticket_mean = df['avg_ticket'].mean()
st.line_chart(df['avg_ticket'], use_container_width=True)

if avg_ticket_recent.mean() > 1.1 * avg_ticket_mean:
    st.warning("⚠️ Average ticket size is significantly above historical average.")
elif avg_ticket_recent.mean() < 0.9 * avg_ticket_mean:
    st.info("ℹ️ Average ticket size is significantly below historical average.")
else:
    st.success("✅ Average ticket size is within normal range.")

# --- Sentiment Analysis ---
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
    return sum(word in text for word in positive_keywords) - sum(word in text for word in negative_keywords)

sentiments = [score_sentiment(h) for h in headlines]
sentiment_score = np.mean(sentiments)

for h, s in zip(headlines, sentiments):
    sentiment_type = "🟢 Positive" if s > 0 else "🔴 Negative" if s < 0 else "🟡 Neutral"
    st.write(f"{sentiment_type}: {h}")

if sentiment_score < -1:
    st.error("⚠️ Negative sentiment detected.")
elif sentiment_score > 1:
    st.success("✅ Headlines suggest positive sentiment.")
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

# --- Interactive Chart ---
st.subheader("Interactive Visualizations")
selected_vars = st.multiselect("Select variables to visualize:", df.columns, default=['revenue', 'store_count'])
st.line_chart(df[selected_vars])

# --- Forecast Plot ---
st.title("Starbucks Revenue Forecasting App")
st.write("This app forecasts Starbucks quarterly revenue using ARIMAX. It uses CPI and store count as predictors.")

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

# --- Risk Flag ---
if risk_flag: 
    st.error("⚠️ Potential Overstatement Risk: Revenue per store exceeds historical range.") 
else: 
    st.success("✅ Revenue per store is within normal historical range.")

# --- AI Summary ---
st.subheader("AI Summary")
st.markdown("""
Based on the ARIMAX forecast using CPI and store count, Starbucks' revenue is projected to remain stable over the next four quarters. 
However, revenue per store shows a potential increase above historical norms. 
This could indicate aggressive revenue projections not matched by store expansion, suggesting a moderate risk of revenue overstatement. 
The average ticket size also appears to be a meaningful signal and should be monitored to better understand consumer behavior trends.
Earnings sentiment and peer comparisons also support the importance of monitoring pricing strategies and external expectations.
""")
