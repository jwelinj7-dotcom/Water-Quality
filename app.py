import streamlit as st
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText

# Load trained model
model = joblib.load("water_quality_model.pkl")
le = joblib.load("label_encoder.pkl")

st.title("Water Quality Monitoring System")

st.write("Enter laboratory test results")

# Input fields
sample_id = st.text_input("Sample ID")

ph = st.number_input("pH Value", 0.0, 14.0, 7.0)
turbidity = st.number_input("Turbidity (NTU)", 0.0, 10.0, 1.0)
hardness = st.number_input("Total Hardness (mg/L)", 0.0, 600.0, 100.0)

user_email = st.text_input("User Email")

# Email sending function


def send_email(receiver_email, result):

    sender_email = "mail"
    password = "app_password"

    message = f"""
WATER QUALITY TEST REPORT

Sample ID: {sample_id}

Measured Parameters:
-----------------------
pH Value: {ph}
Turbidity: {turbidity} NTU
Total Hardness: {hardness} mg/L

Standard Limits:
-----------------------
pH:
Acceptable Limit: 6.5 – 8.5

Turbidity:
Acceptable Limit: 1 NTU
Permissible Limit: 5 NTU

Total Hardness:
Acceptable Limit: 200 mg/L
Permissible Limit: 600 mg/L

Prediction Result:
-----------------------
Water Quality Status: {result}

Note:
If the result is UNSAFE or Moderate consider treatment.

Thank you,
Water Quality Monitoring System
"""

    msg = MIMEText(message)
    msg["Subject"] = "Water Quality Report"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, password)

    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()


# Predict button
if st.button("Analyze Water Sample"):

    sample = pd.DataFrame([[ph, turbidity, hardness]],
                          columns=["pH", "Turbidity", "Total Hardness"])

    prediction = model.predict(sample)

    result = le.inverse_transform(prediction)

    st.success(f"Water Quality: {result[0]}")

    if user_email != "":
        send_email(user_email, result[0])
        st.info("Email alert sent to user")
