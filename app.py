import os


import streamlit as st
import pickle
import numpy as np

st.write("Current directory files:", os.listdir())

# Load model and data
model = pickle.load(open("model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

recommendations = {
    "Cold": "Take warm fluids, rest, and OTC cold medicine.",
    "Flu": "Drink plenty of water and take paracetamol for fever.",
    "COVID": "Isolate, drink fluids, monitor oxygen level, consult doctor.",
    "Allergy": "Avoid allergens and take antihistamines.",
    "Migraine": "Rest in dark room, stay hydrated, avoid stress.",
    "Typhoid": "Take antibiotics only when prescribed by a doctor.",
    "Malaria": "Consult doctor immediately and avoid mosquito bites."
}

st.title("🩺 AI-Based Disease Prediction System")
st.write("Select your symptoms below:")

symptom_inputs = []

user_symptoms = {}
for col in feature_columns:
    user_symptoms[col] = st.checkbox(col)

if st.button("Predict Disease"):
    input_data = np.array([int(user_symptoms[col]) for col in feature_columns]).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    
    st.success(f"Predicted Disease: {predicted_label}")
    st.info(f"Recommendation: {recommendations.get(predicted_label)}")
