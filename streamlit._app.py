import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import joblib


#🌟Title
st.title("💼 Insurance Cost Predictor: A Regression-Based Approach")
st.markdown("""
Welcome to the **Insurance Cost Predictor**, an advanced machine learning tool designed to estimate insurance charges based on personal health and demographic factors.
""")

