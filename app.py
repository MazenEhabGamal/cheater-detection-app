import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load(r'h:\AI\cheater-detection-app\model\cheater_detector.pkl')
scaler = joblib.load(r'h:\AI\cheater-detection-app\model\scaler.pkl')

# App title
st.title('🎯 Cheater Detection in Shooting Games')

# Description
st.write("Enter the player's in-game statistics below to predict whether they are likely to be a cheater or a fair player.")

# Input form
st.subheader('Enter Player Statistics:')
accuracy = st.number_input('🎯 Accuracy (0.0 - 1.0)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
headshot_ratio = st.number_input('💥 Headshot Ratio (0.0 - 1.0)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
reaction_time = st.number_input('⚡ Reaction Time (0.0 - 1.0)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
kd_ratio = st.number_input('🔫 Kill/Death Ratio', min_value=0.0, value=1.0, step=0.1)
shots_per_minute = st.number_input('🔥 Shots Per Minute', min_value=0.0, value=60.0, step=1.0)
avg_kill_distance = st.number_input('📏 Average Kill Distance', min_value=0.0, value=40.0, step=1.0)
win_rate = st.number_input('🏆 Win Rate (0.0 - 1.0)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Predict button
if st.button('Detect Cheater'):
    input_data = np.array([[accuracy, headshot_ratio, reaction_time, kd_ratio,
                            shots_per_minute, avg_kill_distance, win_rate]])
    
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(input_scaled)[0]
    prediction_prob = model.predict_proba(input_scaled)[0][1]  # Probability of being a cheater
    
    if prediction == 1:
        st.error(f'🚨 This player is likely a CHEATER with a probability of {prediction_prob:.2%}.')
    else:
        st.success(f'✅ This player seems to be FAIR with a probability of {1 - prediction_prob:.2%}.')

# Footer
st.markdown("---")
st.caption("Developed by Mazen Ehab | AI-Powered Cheater Detection")
