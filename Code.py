import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from pandas_datareader import data as pdr 
from datetime import datetime 
import warnings 

warnings.filterwarnings("ignore")


df = pd.read_csv("starbucks_financials_expanded.csv") 
df['date'] = pd.to_datetime(df['date']) 
df.set_index('date', inplace=True) 
df = df.asfreq('Q')


# --- Sidebar: CPI input toggle ---
st.sidebar.header("CPI Input Options") 
use_live_cpi = st.sidebar.checkbox("Use Live CPI from FRED", value=True)

if use_live_cpi: 
    try:
        live_cpi = pdr.get_data_fred('CPIAUCSL', start=df.index.min(), end=datetime.today()) 
        live_cpi = live_cpi.resample('Q').mean()
        df['CPI'] = live_cpi['CPIAUCSL'].reindex(df.index).fillna(method='ffill') 
        st.sidebar.success("✅ Live CPI data fetched successfully.") 
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to fetch live CPI: {e}") 
        use_live_cpi = False

if not use_live_cpi: 
    cpi_manual = st.sidebar.number_input(
        "Enter CPI value for forecast period", 
        min_value=0.0, max_value=500.0, value=300.0
    )
    df['CPI'].iloc[-4:] = cpi_manual  # Override last 4 quarters

# --- Forecast revenue using ARIMAX ---
revenue = df['revenue'] 
exog = df[['CPI', 'store_count']]

train_revenue = revenue[:-4] 
test_revenue = revenue[-4:] 
train_exog = exog[:-4] 
test_exog = exog[-4:]

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

# --- Plot ---
st.title("Starbucks Revenue Forecasting App")
st.write("""
This app forecasts Starbucks quarterly revenue using ARIMAX. 
It incorporates store count and CPI as predictors. Users can choose to use live CPI data from FRED or enter manual CPI values.
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
Continued monitoring of store growth and macroeconomic conditions is advised.
""")
