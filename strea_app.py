import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Title
st.set_page_config(page_title="Salary Prediction App")
st.title("💼 Salary Prediction App")
st.write("This app predicts salary category based on user inputs.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\ganes\Downloads\adult 3.csv")
    df.dropna(inplace=True)
    return df

# Encode categorical features
def encode_features(df):
    encoders = {}
    for column in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoders[column] = le
    return df, encoders

# Load and preprocess
data = load_data()
data, label_encoders = encode_features(data)

# Split features and target
X = data.drop('income', axis=1)  # Replace 'salary' if different
y = data['income']
model = RandomForestClassifier()
model.fit(X, y)

# Input form
st.header("📋 Enter Your Details")
user_input = {}
for col in X.columns:
    if col in label_encoders:
        options = list(label_encoders[col].classes_)
        user_input[col] = st.selectbox(col, options)
    else:
        user_input[col] = st.number_input(col, value=0)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Encode user inputs
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Predict button
if st.button("🎯 Predict Salary Category"):
    pred = model.predict(input_df)
    pred_label = label_encoders['income'].inverse_transform(pred)
    st.success(f"✅ Predicted Salary Category: **{pred_label[0]}**")