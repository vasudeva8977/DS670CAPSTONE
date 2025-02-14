import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
model = joblib.load('fraud_detection_model.joblib')

# Define the app UI
st.set_page_config(page_title='Fraud Detection App', layout='wide')
st.title('🔍 Fraud Detection App')
st.markdown('**Predict fraudulent transactions based on input data.**')

# Sidebar
st.sidebar.header('📝 Input Transaction Details')
st.sidebar.markdown('Fill in the details below to check for fraud.')

# Example data
column_data_types = {
    'Reference Number': 'int64',
    'Control-Number': 'float64',
    'Financial-Institution-Number': 'int64',
    'Deposit-Business-Date': 'datetime64[ns]',
    'Financial-Institution-Business-Date': 'datetime64[ns]',
    'Financial-Institution-Transaction-Date-Date': 'datetime64[ns]',
    'Financial-Institution-Transaction-Type-Code': 'object',
    'Financial-Institution-Transaction-Amount': 'float64',
    'Authorization-Number': 'object',
    'Transaction-Amount-Deviation': 'int64'
}

# User input fields
input_data = {}
st.sidebar.subheader('📌 Enter Transaction Data')
for column, dtype in column_data_types.items():
    if dtype == 'object':
        input_value = st.sidebar.text_input(f'🔤 {column}', value='Enter value')
    elif dtype == 'float64':
        input_value = st.sidebar.number_input(f'💰 {column}', value=0.0)
    elif dtype == 'int64':
        input_value = st.sidebar.number_input(f'🔢 {column}', value=0)
    input_data[column] = input_value

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Drop unused or date columns before prediction
drop_columns = ["Authorization-Number", "Deposit-Business-Date", 
                "Financial-Institution-Business-Date", "Financial-Institution-Transaction-Date-Date", 
                "Financial-Institution-Transaction-Type-Code"]
input_df.drop(columns=[col for col in drop_columns if col in input_df.columns], inplace=True, errors='ignore')

# Load expected training features
expected_features = model.feature_names_in_

# One-hot encode categorical features
input_df = pd.get_dummies(input_df)

# Ensure all features match training data
for feature in expected_features:
    if feature not in input_df.columns:
        input_df[feature] = 0  # Add missing features as zero

# Reorder columns to match training data
input_df = input_df[expected_features]

# Convert all input columns to float before prediction
input_df = input_df.astype(float)

# Prediction button
if st.sidebar.button('🔮 Predict Fraud'):
    prediction = model.predict(input_df)
    st.markdown("## Prediction Result")
    if prediction[0] == 1:
        st.error('🚨 **Fraudulent Transaction Detected!**')
    else:
        st.success('✅ **Transaction is NOT Fraudulent.**')

# Add a simple data visualization
st.markdown("## 📊 Data Insights")
st.write("A simple overview of fraud detection trends.")
example_data = pd.DataFrame({
    "Transaction Amount": np.random.randint(100, 10000, 50),
    "Fraud Status": np.random.choice([0, 1], 50, p=[0.8, 0.2])
})
fig, ax = plt.subplots()
sns.histplot(example_data, x="Transaction Amount", hue="Fraud Status", bins=20, kde=True, ax=ax)
st.pyplot(fig)

st.markdown("---")
st.markdown("🚀 **Built with Streamlit & Machine Learning for Fraud Detection**")
