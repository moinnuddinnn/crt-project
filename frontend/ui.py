import streamlit as st
import requests

st.set_page_config(
    page_title="Student Assignment Completion Time Predictor",
    layout="centered"
)

st.title("📚 Assignment Completion Time Predictor")

st.markdown("""
Estimate how long a student will take to complete an assignment  
based on **difficulty** and **focus level**.
""")

# Inputs
difficulty = st.slider("Difficulty Level", 1, 10, 5)
focus = st.slider("Focus Level", 1, 10, 5)

# Prediction
if st.button("Predict Completion Time"):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={
                "difficulty_level": difficulty,
                "focus_level": focus
            },
            timeout=5
        )

        result = response.json()

        st.success(
            f"Estimated Time: **{result['predicted_completion_time']} minutes**"
        )

    except:
        st.error("⚠️ API not running. Start FastAPI server.")

st.markdown("---")
st.caption("Built with Linear Regression + Streamlit")