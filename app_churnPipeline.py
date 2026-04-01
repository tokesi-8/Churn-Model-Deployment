import streamlit as st
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join("artifacts", "churn_prediction_pipeline.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def main():
    st.title('Churn Model Deployment')

    age = st.number_input("age", 0, 100)
    gender = st.radio("gender", ["Male","Female"])
    tenure = st.number_input("tenure", 0,100)
    usage_freq = st.number_input("usage frequency", 0,100)
    support_call = st.number_input("support calls", 0,10)
    payment_delay = st.number_input("payment delay", 0,30)
    subs_type = st.radio("subscription type", ["Standard","Premium","Basic"])
    contract_length = st.radio("contract length", ["Annual","Quarterly","Monthly"])
    total_spend = st.number_input("total spend", 0,1000000000)
    last_interaction = st.number_input("last interaction", 0,30)

    if st.button("Make Prediction"):
        data = {
            'Age': age,
            'Gender': gender,
            'Tenure': tenure,
            'UsageFrequency': usage_freq,
            'SupportCalls': support_call,
            'PaymentDelay': payment_delay,
            'SubscriptionType': subs_type,
            'ContractLength': contract_length,
            'TotalSpend': total_spend,
            'LastInteraction': last_interaction
        }

        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]

        st.success(f"Churn Prediction: {prediction}")

if __name__ == "__main__":
    main()
