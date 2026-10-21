import streamlit as st

import eda
import prediction

st.set_page_config(
    page_title="Heart Disease Screening",
    page_icon="❤️",
    layout="wide",
)

st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih halaman", ["Prediksi", "EDA"])
st.sidebar.caption(
    "Aplikasi skrining awal penyakit jantung (model SVM). "
    "Bukan pengganti diagnosis dokter."
)

if page == "Prediksi":
    prediction.run()
else:
    eda.run()
