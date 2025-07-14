import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model (assume you saved RandomForest model as 'ckd_model.pkl')
model = pickle.load(open("ckd_model.pkl", "rb"))

# UI title
st.title("Chronic Kidney Disease Detector")

# Define user input fields
st.subheader("Enter patient data:")

age = st.number_input("Age", min_value=0, max_value=120)
bp = st.number_input("Blood Pressure (bp)")
sg = st.selectbox("Specific Gravity (sg)", [1.005, 1.010, 1.015, 1.020, 1.025])
al = st.slider("Albumin", 0, 5)
su = st.slider("Sugar", 0, 5)
rbc = st.selectbox("Red Blood Cells (rbc)", ["normal", "abnormal"])
pc = st.selectbox("Pus Cell (pc)", ["normal", "abnormal"])
pcc = st.selectbox("Pus Cell Clumps (pcc)", ["present", "notpresent"])
ba = st.selectbox("Bacteria (ba)", ["present", "notpresent"])
bgr = st.number_input("Blood Glucose Random (bgr)")
bu = st.number_input("Blood Urea (bu)")
sc = st.number_input("Serum Creatinine (sc)")
sod = st.number_input("Sodium (sod)")
pot = st.number_input("Potassium (pot)")
hemo = st.number_input("Hemoglobin (hemo)")
pcv = st.number_input("Packed Cell Volume (pcv)")
wc = st.number_input("White Blood Cell Count (wc)")
rc = st.number_input("Red Blood Cell Count (rc)")
htn = st.selectbox("Hypertension (htn)", ["yes", "no"])
dm = st.selectbox("Diabetes Mellitus (dm)", ["yes", "no"])
cad = st.selectbox("Coronary Artery Disease (cad)", ["yes", "no"])
appet = st.selectbox("Appetite", ["good", "poor"])
pe = st.selectbox("Pedal Edema (pe)", ["yes", "no"])
ane = st.selectbox("Anemia (ane)", ["yes", "no"])

# Encode categorical features as the model expects numerical input
def encode(val):
    return LabelEncoder().fit(["abnormal", "normal"]).transform([val])[0]

rbc = encode(rbc)
pc = encode(pc)
pcc = 1 if pcc == "present" else 0
ba = 1 if ba == "present" else 0
htn = 1 if htn == "yes" else 0
dm = 1 if dm == "yes" else 0
cad = 1 if cad == "yes" else 0
appet = 1 if appet == "good" else 0
pe = 1 if pe == "yes" else 0
ane = 1 if ane == "yes" else 0

# Prepare input for model
features = np.array([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot,
                      hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]])

# Predict and display result
if st.button("Detect"):
    prediction = model.predict(features)
    if prediction[0] == 0:
        st.error("The person is likely to have Chronic Kidney Disease.")
    else:
        st.success("The person is NOT likely to have Chronic Kidney Disease.")