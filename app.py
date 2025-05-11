import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing artifacts
model = joblib.load("xgb_conversion_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title(" Customer Conversion Predictor")
st.markdown("Predict whether a customer will convert based on marketing attributes.")

# Input form
with st.form("conversion_form"):
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
    income = st.number_input("Income", 0, 1_000_000, 45000)
    campaign_channel = st.selectbox("Campaign Channel", label_encoders['CampaignChannel'].classes_)
    campaign_type = st.selectbox("Campaign Type", label_encoders['CampaignType'].classes_)
    ad_spend = st.number_input("Ad Spend", 0.0, 1_000_000.0, 150.0)
    ctr = st.slider("Click Through Rate (CTR)", 0.0, 1.0, 0.05)
    conv_rate = st.slider("Conversion Rate", 0.0, 1.0, 0.02)
    visits = st.number_input("Website Visits", 0, 1000, 12)
    pages_per_visit = st.number_input("Pages Per Visit", 0.0, 100.0, 3.5)
    time_on_site = st.number_input("Time on Site (minutes)", 0.0, 100.0, 6.8)
    social_shares = st.number_input("Social Shares", 0, 100, 2)
    email_opens = st.number_input("Email Opens", 0, 100, 4)
    email_clicks = st.number_input("Email Clicks", 0, 100, 1)
    purchases = st.number_input("Previous Purchases", 0, 100, 1)
    loyalty_points = st.number_input("Loyalty Points", 0, 10000, 120)
    ad_platform = st.selectbox("Advertising Platform", label_encoders['AdvertisingPlatform'].classes_)
    ad_tool = st.selectbox("Advertising Tool", label_encoders['AdvertisingTool'].classes_)

    submitted = st.form_submit_button("Predict Conversion")

if submitted:
    # Build input dataframe
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Income': income,
        'CampaignChannel': campaign_channel,
        'CampaignType': campaign_type,
        'AdSpend': ad_spend,
        'ClickThroughRate': ctr,
        'ConversionRate': conv_rate,
        'WebsiteVisits': visits,
        'PagesPerVisit': pages_per_visit,
        'TimeOnSite': time_on_site,
        'SocialShares': social_shares,
        'EmailOpens': email_opens,
        'EmailClicks': email_clicks,
        'PreviousPurchases': purchases,
        'LoyaltyPoints': loyalty_points,
        'AdvertisingPlatform': ad_platform,
        'AdvertisingTool': ad_tool
    }])

    # Encode categorical features
    for col, le in label_encoders.items():
        input_data[col] = le.transform(input_data[col])

    # Scale features
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # Display results
    st.subheader(" Prediction Result")
    if prediction == 1:
        st.success(f" Customer is likely to CONVERT (Probability: {probability:.2f})")
    else:
        st.error(f" Customer is NOT likely to convert (Probability: {probability:.2f})")