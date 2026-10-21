from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

DATA_PATH = Path(__file__).resolve().parent / "heart.csv"

NUM_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CAT_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope", "FastingBS"]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # nilai 0 pada kedua kolom ini mustahil secara medis -> dianggap kosong
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
    df["RestingBP"] = df["RestingBP"].replace(0, np.nan)
    return df


def run():
    st.title("Exploratory Data Analysis")
    df = load_data()

    tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Numerik", "Kategorikal", "Korelasi"])

    with tab1:
        st.write(f"Dataset: **{df.shape[0]} baris**, **{df.shape[1]} kolom**.")
        st.dataframe(df.head(10))
        c1, c2 = st.columns(2)
        with c1:
            st.write("Distribusi target (HeartDisease)")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.countplot(data=df, x="HeartDisease", ax=ax)
            st.pyplot(fig)
        with c2:
            st.write("Missing value (0 pada Cholesterol dan RestingBP dihitung kosong)")
            st.dataframe(df.isna().sum().rename("jumlah_kosong"))

    with tab2:
        col = st.selectbox("Pilih kolom numerik", NUM_COLS)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.histplot(data=df, x=col, hue="HeartDisease", kde=True, bins=20, ax=axes[0])
        sns.boxplot(data=df, x="HeartDisease", y=col, ax=axes[1])
        axes[0].set_title(f"Distribusi {col}")
        axes[1].set_title(f"{col} per kelas")
        st.pyplot(fig)
        st.caption(f"Skewness {col}: {df[col].skew():.2f}")

    with tab3:
        col = st.selectbox("Pilih kolom kategorikal", CAT_COLS)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x=col, hue="HeartDisease", ax=ax)
        ax.set_title(f"{col} vs HeartDisease")
        st.pyplot(fig)

        rows = []
        for c in CAT_COLS:
            p = stats.chi2_contingency(pd.crosstab(df[c], df["HeartDisease"])).pvalue
            rows.append({"fitur": c, "p_value": p, "signifikan (p<0.05)": p < 0.05})
        st.write("Uji Chi-Squared terhadap target")
        st.dataframe(pd.DataFrame(rows).sort_values("p_value"))

    with tab4:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(df[NUM_COLS + ["HeartDisease"]].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        st.caption("Korelasi dengan target bersifat moderat; hubungan ini bukan sebab-akibat.")
