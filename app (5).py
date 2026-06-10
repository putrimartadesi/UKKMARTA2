import streamlit as st
import pandas as pd
import pickle

# ======================
# LOAD MODEL
# ======================
try:
    with open("mental_health_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ======================
# TAMPILAN
# ======================
st.set_page_config(
    page_title="Prediksi Kesehatan Mental",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Prediksi Kesehatan Mental Mahasiswa")

# ======================
# INPUT USER
# ======================
gpa = st.slider("GPA", 0.0, 4.0, 3.0)

stress = st.slider(
    "Stress Level",
    0,
    10,
    5
)

anxiety = st.slider(
    "Anxiety Score",
    0,
    10,
    5
)

depression = st.slider(
    "Depression Score",
    0,
    10,
    5
)

mood = st.selectbox(
    "Mood Description",
    [
        "Happy",
        "Calm",
        "Neutral",
        "Sad",
        "Anxious",
        "Stressed"
    ]
)

daily_reflections = st.number_input(
    "Daily Reflections",
    min_value=0,
    max_value=100,
    value=50
)

# ======================
# ENCODING MOOD
# ======================
mood_map = {
    "Happy": 0,
    "Calm": 1,
    "Neutral": 2,
    "Sad": 3,
    "Anxious": 4,
    "Stressed": 5
}

# ======================
# DATA INPUT
# ======================
input_df = pd.DataFrame({
    "GPA": [gpa],
    "Stress_Level": [stress],
    "Anxiety_Score": [anxiety],
    "Depression_Score": [depression],
    "Mood_Description": [mood_map[mood]],
    "Daily_Reflections": [daily_reflections]
})

# ======================
# CEK FITUR MODEL
# ======================
if hasattr(model, "feature_names_in_"):

    fitur_model = list(model.feature_names_in_)

    for kolom in fitur_model:
        if kolom not in input_df.columns:
            input_df[kolom] = 0

    input_df = input_df[fitur_model]

st.subheader("Data Input")
st.dataframe(input_df)

# ======================
# PREDIKSI
# ======================
if st.button("Prediksi"):

    try:

        data_scaled = scaler.transform(input_df)

        hasil = model.predict(data_scaled)[0]

        st.success(f"Hasil Prediksi: {hasil}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ======================
# DEBUG FITUR MODEL
# ======================
with st.expander("Informasi Model"):

    if hasattr(model, "feature_names_in_"):
        st.write("Fitur yang digunakan model:")
        st.write(list(model.feature_names_in_))
