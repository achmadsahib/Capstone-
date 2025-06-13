import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model
with open('best_model_rf.pkl', 'rb') as file:
    model = pickle.load(file)

# Load scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# --- Untuk encoding kategori (misal label encoding manual) ---
def encode_gender(g):
    return 1 if g == "Male" else 0

def encode_yes_no(val):
    return 1 if val == "yes" else 0

def encode_CAEC(val):
    mapping = {"no":0, "Sometimes":1, "Frequently":2, "Always":3}
    return mapping.get(val, 0)

def encode_CALC(val):
    mapping = {"no":0, "Sometimes":1, "Frequently":2, "Always":3}
    return mapping.get(val, 0)

def encode_MTRANS(val):
    mapping = {
        "Public_Transportation":0,
        "Walking":1,
        "Automobile":2,
        "Motorbike":3,
        "Bike":4
    }
    return mapping.get(val, 0)

# Judul aplikasi
st.title("Prediksi Kategori Obesitas")
st.write("Masukkan data diri Anda untuk memprediksi status obesitas menggunakan model machine learning.")

# Input user
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.slider("Usia", 10, 100, 25)
height = st.number_input("Tinggi badan (meter)", value=1.70, step=0.01)
weight = st.number_input("Berat badan (kg)", value=65.0, step=0.5)
family_history = st.selectbox("Riwayat keluarga dengan obesitas?", ["yes", "no"])
FAVC = st.selectbox("Sering mengonsumsi makanan tinggi kalori?", ["yes", "no"])
FCVC = st.slider("Frekuensi konsumsi sayur (0-3)", 0.0, 3.0, 2.0)
NCP = st.slider("Jumlah kali makan per hari", 1.0, 4.0, 3.0)
CAEC = st.selectbox("Konsumsi makanan di antara waktu makan?", ["no", "Sometimes", "Frequently", "Always"])
SMOKE = st.selectbox("Merokok?", ["yes", "no"])
CH2O = st.slider("Konsumsi air harian (liter)", 0.0, 3.0, 2.0)
SCC = st.selectbox("Apakah Anda memantau asupan kalori harian Anda?", ["yes", "no"])
FAF = st.slider("Aktivitas fisik (jam/minggu)", 0.0, 5.0, 1.0)
TUE = st.slider("Waktu penggunaan teknologi (jam/hari)", 0.0, 5.0, 2.0)
CALC = st.selectbox("Konsumsi alkohol?", ["no", "Sometimes", "Frequently", "Always"])
MTRANS = st.selectbox("Transportasi utama", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

if st.button("Prediksi"):
    # Encode dan transformasi input
    input_data = np.array([[
        encode_gender(gender),
        age,
        height,
        weight,
        encode_yes_no(family_history),
        encode_yes_no(FAVC),
        FCVC,
        NCP,
        encode_CAEC(CAEC),
        encode_yes_no(SMOKE),
        CH2O,
        encode_yes_no(SCC),
        FAF,
        TUE,
        encode_CALC(CALC),
        encode_MTRANS(MTRANS)
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Mapping prediksi ke label
    label_mapping = {
        1: "Underweight",
        2: "Normal weight",
        3: "Overweight",
        4: "Obesity Type I",
        5: "Obesity Type II",
        6: "Obesity Type III"
    }
    predicted_label = label_mapping.get(prediction, "Unknown")

    st.success(f"Hasil prediksi: **{predicted_label}**")

