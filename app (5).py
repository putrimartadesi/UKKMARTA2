import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("mental_health_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Kesehatan Mental Mahasiswa")

# Input
gpa = st.slider("GPA", 1.0, 4.0, 3.0)
stress = st.slider("Stress Level", 0, 10, 5)
anxiety = st.slider("Anxiety Score", 0, 10, 5)
depression = st.slider("Depression Score", 0, 10, 5)

mood = st.selectbox(
    "Mood Description",
    ["Happy","Calm","Neutral","Sad","Anxious","Stressed","Motivated","Frustrated"]
)

daily_reflections = st.text_input(
    "Daily Reflections",
    "I feel good today"
)

mood_map = {
    "Happy":0,
    "Calm":1,
    "Neutral":2,
    "Sad":3,
    "Anxious":4,
    "Stressed":5,
    "Motivated":6,
    "Frustrated":7
}

# Data input
data = {
    "GPA": gpa,
    "Stress_Level": stress,
    "Anxiety_Score": anxiety,
    "Depression_Score": depression,
    "Mood_Description": mood_map[mood],
    "Daily_Reflections": daily_reflections
}

input_df = pd.DataFrame([data])

# PENTING:
# Sesuaikan urutan kolom dengan model
if hasattr(model, "feature_names_in_"):
    input_df = input_df.reindex(columns=model.feature_names_in_)

st.write("Data yang dikirim ke model:")
st.dataframe(input_df)

if st.button("Prediksi"):
    try:

        X = scaler.transform(input_df)

        hasil = model.predict(X)[0]

        if hasil == 0:
            st.success("Healthy 😊")

        elif hasil == 1:
            st.warning("At Risk ⚠️")

        else:
            st.error("Struggling 🚨")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
