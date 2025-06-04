import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from pandas_datareader import data as pdr 
from datetime import datetime 
import warnings 

st.title("Starbucks Revenue Forecasting App")

st.markdown("""
Welcome to the Starbucks Revenue Forecasting App!  
This application provides a powerful tool to forecast Starbucks' quarterly revenue using advanced time-series modeling. 
By leveraging economic indicators like CPI and operational metrics like store count, the app delivers insights into future revenue trends. 
Explore interactive visualizations, analyze market sentiment, and benchmark performance against industry peers to make informed decisions.
""")
