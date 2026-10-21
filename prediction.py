import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from feature_engine.outliers import Winsorizer  # noqa: F401  (wajib ada agar pickle bisa dibaca)

MODEL_PATH = Path(__file__).resolve().parent / "model_svm.pkl"

CHEST_PAIN = {
    "ASY - Tanpa gejala nyeri dada": "ASY",
    "ATA - Atypical angina (kurang khas)": "ATA",
    "NAP - Non-anginal pain (bukan dari jantung)": "NAP",
    "TA - Typical angina (khas)": "TA",
}
RESTING_ECG = {
    "Normal": "Normal",
    "ST - Kelainan gelombang ST-T": "ST",
    "LVH - Kemungkinan pembesaran bilik kiri": "LVH",
}
ST_SLOPE = {
    "Up - Naik": "Up",
    "Flat - Datar": "Flat",
    "Down - Turun": "Down",
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_input(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    """Susun 1 baris DataFrame dengan kolom persis seperti saat training.

    Nilai yang tidak diketahui WAJIB np.nan (bukan 0), supaya imputer di
    dalam pipeline mengisinya dengan median data train.
    """
    return pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }])


def run():
    st.title("Skrining Awal Penyakit Jantung")
    st.write(
        "Isi data klinis pasien lalu klik **Prediksi**. Hasil ini hanya alat bantu "
        "skrining awal untuk menentukan perlu tidaknya pemeriksaan lanjutan."
    )

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("File model_svm.pkl tidak ditemukan di folder yang sama dengan streamlit_app.py.")
        return

    with st.form("form_pasien"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.slider("Usia (tahun)", 20, 90, 55)
            sex = st.selectbox("Jenis kelamin", ["M", "F"],
                               format_func=lambda x: "Laki-laki" if x == "M" else "Perempuan")
            chest_label = st.selectbox("Jenis nyeri dada", list(CHEST_PAIN))
            resting_bp = st.number_input("Tekanan darah istirahat (mm Hg)", 80, 220, 130)
            bp_unknown = st.checkbox("Tekanan darah tidak diketahui")
            cholesterol = st.number_input("Kolesterol (mg/dl)", 100, 610, 220)
            chol_unknown = st.checkbox("Kolesterol tidak diketahui / belum diperiksa")
            fasting_bs = st.selectbox("Gula darah puasa > 120 mg/dl?", [0, 1],
                                      format_func=lambda x: "Ya" if x == 1 else "Tidak")

        with c2:
            ecg_label = st.selectbox("Hasil EKG istirahat", list(RESTING_ECG))
            max_hr = st.slider("Detak jantung maksimum", 60, 210, 140)
            exercise_angina = st.selectbox("Nyeri dada saat olahraga?", ["N", "Y"],
                                           format_func=lambda x: "Ya" if x == "Y" else "Tidak")
            oldpeak = st.number_input("Oldpeak (penurunan segmen ST)", -3.0, 6.5, 0.0, step=0.1)
            slope_label = st.selectbox("Kemiringan segmen ST (ST_Slope)", list(ST_SLOPE))

        submitted = st.form_submit_button("Prediksi")

    if not submitted:
        return

    df_input = build_input(
        age=age,
        sex=sex,
        chest_pain=CHEST_PAIN[chest_label],
        resting_bp=np.nan if bp_unknown else resting_bp,
        cholesterol=np.nan if chol_unknown else cholesterol,
        fasting_bs=fasting_bs,
        resting_ecg=RESTING_ECG[ecg_label],
        max_hr=max_hr,
        exercise_angina=exercise_angina,
        oldpeak=oldpeak,
        st_slope=ST_SLOPE[slope_label],
    )

    pred = int(model.predict(df_input)[0])
    score = float(model.decision_function(df_input)[0])

    st.subheader("Hasil")
    if pred == 1:
        st.error("**Berisiko penyakit jantung.** Disarankan pemeriksaan lanjutan.")
    else:
        st.success("**Tidak terdeteksi risiko penyakit jantung** berdasarkan data yang dimasukkan.")

    st.caption(
        f"Skor keputusan SVM: {score:.2f} (di atas 0 = kelas sakit jantung; "
        "ini bukan probabilitas)."
    )
    if bp_unknown or chol_unknown:
        st.info("Nilai yang tidak diketahui diisi otomatis dengan median data latih.")

    with st.expander("Data yang dikirim ke model"):
        st.dataframe(df_input)
