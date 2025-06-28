
import streamlit as st
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Geological Ops Assistant", layout="wide")
st.title("🛢 Geological Operations Assistant")

# Sample employee data
data = {
    "Employee ID": [
        "OPC00425", "OPC00426", "OPC01066", "OPC00417", "OPC00418",
        "OPC00419", "OPC00420", "OPC00421", "OPC00422", "OPC00423",
        "OPC00424", "OPC00427", "OPC00428", "OPC00429", "OPC00430",
        "OPC00431", "OPC00432"
    ],
    "Employee Name": [
        "Mohamed Salih Eidam Adam", "Ibrahim Abdel Aziz Mohamed", "Ali Ibrahim Gulfan",
        "Sami Mohamed Elfatih Saad Ahmed", "Hassan Eltayeb Hassan Mohammed",
        "Omer Ibrahim Omer", "Mansour Adam Mansour", "Ahmed Hassan Khalil",
        "Tariq Omer Ahmed", "Yasir Elhadi Mohammed", "Nour Eldin Ibrahim",
        "Osman Ahmed Osman", "Abdelrahman Ismail", "Salah Ahmed Abdelrahman",
        "Hassan Idris Ali", "Mohamed Ahmed Elamin", "Osman Khalid Bashir"
    ],
    "Position": [
        "Geological Superintendent", "Geological Superintendent", "Wellsite Geological Supervisor",
        "Wellsite Geological Supervisor", "Wellsite Geological Supervisor", "Wellsite Geological Supervisor",
        "Wellsite Geological Supervisor", "Wellsite Geological Supervisor", "Wellsite Geological Supervisor",
        "Operations Geologist", "Operations Geologist", "Pore Pressure Engineer", "Pore Pressure Engineer",
        "Wellsite Geological Supervisor", "Wellsite Geological Supervisor", "Geological Superintendent",
        "Geological Superintendent"
    ],
    "Email": [
        "meidam@2bopco.com", "iaziz@2bopco.com", "agulfan@gnpoc.com", "smelfatih@2bopco.com",
        "hehassan@2bopco.com", "oiomer@2bopco.com", "mamansour@2bopco.com", "ahkhalil@2bopco.com",
        "toahmed@2bopco.com", "yemohammed@2bopco.com", "neibrahim@2bopco.com", "oaosman@2bopco.com",
        "aismail@2bopco.com", "saahmed@2bopco.com", "hiali@2bopco.com", "maelamin@2bopco.com",
        "okbashir@2bopco.com"
    ]
}

# Prepare data
df = pd.DataFrame(data)
df["Email Domain"] = df["Email"].apply(lambda x: x.split("@")[-1])
df["Name_Length"] = df["Employee Name"].apply(len)
df["ID_Num"] = df["Employee ID"].str.extract("(\d+)").astype(float)
df["Domain_ID"] = LabelEncoder().fit_transform(df["Email Domain"])
label_encoder = LabelEncoder()
df["Position_Label"] = label_encoder.fit_transform(df["Position"])

# Train model
features = df[["Name_Length", "ID_Num", "Domain_ID"]]
target = df["Position_Label"]
model = RandomForestClassifier(random_state=42)
model.fit(features, target)

# Tabs for navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Role Predictor", "📧 Email Validator", "🤖 Auto-fill Assistant", 
    "📈 Upload + Reports", "📉 Attrition (Coming Soon)"
])

# Role Predictor
with tab1:
    st.header("🔍 Predict Geological Role")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Employee Name", "John Smith")
        emp_id = st.text_input("Employee ID", "EMP000")
    with col2:
        email = st.text_input("Email", "jsmith@2bopco.com")

    if st.button("Predict Role"):
        name_len = len(name)
        id_num = float(re.search(r"(\d+)", emp_id).group(1)) if re.search(r"(\d+)", emp_id) else 0.0
        domain_id = 0 if "2bopco.com" in email else 1
        features = [[name_len, id_num, domain_id]]
        pred = model.predict(features)[0]
        role = label_encoder.inverse_transform([pred])[0]
        st.success(f"🧠 Predicted Role: **{role}**")

    st.subheader("📊 Position Distribution")
    st.bar_chart(df["Position"].value_counts())

# Email Validator
with tab2:
    st.header("📧 Email Domain Validator")
    email_input = st.text_input("Enter an email to check")
    if st.button("Check Domain"):
        domain = email_input.split("@")[-1]
        if domain in df["Email Domain"].unique():
            st.success(f"✅ Valid domain: {domain}")
        else:
            st.error(f"❌ Suspicious or unknown domain: {domain}")

# Auto-fill Assistant
with tab3:
    st.header("🤖 Smart Auto-fill Assistant")
    emp_id_partial = st.text_input("Start typing Employee ID", "OPC")
    if emp_id_partial.startswith("OPC"):
        st.info("✅ ID format recognized")
        st.write("Suggested Email: `user@2bopco.com`")

# CSV Upload + Summary
with tab4:
    st.header("📈 Upload CSV for Summary Report")
    uploaded_file = st.file_uploader("Upload employee dataset (CSV)", type=["csv"])
    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("Preview:")
        st.dataframe(df_uploaded.head())
        st.write("📊 Summary Statistics:")
        st.write(df_uploaded.describe(include='all'))

# Attrition Predictor
with tab5:
    st.header("📉 Employee Attrition Predictor (Coming Soon)")
    st.info("This feature will predict employee retention risk using HR indicators.")

