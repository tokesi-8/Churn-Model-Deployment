import streamlit as st
import joblib
import pandas as pd
import os

@st.cache_resource
def load_model():
    path = os.path.join("artifacts", "churn_prediction_pipeline.pkl")
    return joblib.load(path)

def main():
    st.title('Churn Model Deployment')

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error load model: {e}")
        return

    age = st.number_input("age", 0, 100)
    gender = st.radio("gender", ["Male","Female"])

    if st.button("Test Model"):
        st.success("Model berhasil load!")

if __name__ == "__main__":
    main()
