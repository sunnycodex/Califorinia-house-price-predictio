import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('decision_tree_regressor_model.joblib')

st.title("California Housing Price Prediction")

# Feature names from the California housing dataset
feature_names = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

# Create input fields for each feature
user_input = []
for feature in feature_names:
    value = st.number_input(f"Enter value for {feature}:", value=0.0)
    user_input.append(value)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)
    st.success(f"Predicted House Price: {prediction[0]:.2f}")

st.markdown("""
*Model: Decision Tree Regressor (GridSearchCV tuned)*  
*All features are required. Use values similar to the dataset for best results.*
""")