import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

API_KEY = "e1fc84d16d5a8a68b7d600d4aa49423b"  # 🔒 better to keep hidden later

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()

    if data.get("cod") != 200:
        return None

    return {
        "TMPC": data["main"]["temp"],
        "RELH": data["main"]["humidity"],
        "SPED": data["wind"]["speed"],
        "VSBK": data.get("visibility", 10000) / 1000
    }


def render_ui(model, feature_cols, df, train_acc, test_acc):

    st.set_page_config(layout="wide")
    st.title("🌩️ AI Storm Prediction + Explainable AI")

    st.sidebar.header("⚙️ Mode")
    mode = st.sidebar.radio("Select", ["Live Weather", "Auto Data"])

    # -------------------------------
    # Accuracy Display
    # -------------------------------
    st.sidebar.markdown("###Model Accuracy")
    st.sidebar.write(f"Train: {train_acc:.2f}")
    st.sidebar.write(f"Test: {test_acc:.2f}")

    # -------------------------------
    # INPUT
    # -------------------------------
    input_df = None  # ✅ FIX

    if mode == "Live Weather":
        city = st.text_input("City", "Kolkata")

        if st.button(" Get Weather & Predict"):

            weather = get_weather(city)

            if weather is None:
                st.error("❌ API Error or Invalid City")
                return

            input_df = pd.DataFrame([weather])

    else:
        input_data = {
            col: float(df[col].sample(1).values[0])
            for col in feature_cols
        }
        input_df = pd.DataFrame([input_data])

    # -------------------------------
    # SAFETY CHECK
    # -------------------------------
    if input_df is None:
        st.warning("⚠️ Click the button to fetch weather data")
        return

    # Fill missing columns
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = df[col].mean()

    input_df = input_df[feature_cols]

    # -------------------------------
    # SHOW INPUT
    # -------------------------------
    st.subheader(" Input Data")
    st.write(input_df)

    # -------------------------------
    # 🔥 PREDICTION (AUTO)
    # -------------------------------
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Storm Likely")
    else:
        st.success("✅ No Storm")

    # -------------------------------
    # 📊 FEATURE IMPORTANCE
    # -------------------------------
    st.subheader("Graph")

    importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(feature_cols, importances)
    st.pyplot(fig)

    # -------------------------------
    # 🧠 LOCAL EXPLANATION
    # -------------------------------
    st.subheader(" values of the coloumns ")

    important_features = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    for feature, score in important_features:
        value = input_df.iloc[0][feature]
        st.write(f"{feature} = {value:.2f} → influence: {score:.2f}")

    # -------------------------------
    # 📜 RULE EXPLANATION
    # -------------------------------
    st.subheader(" Rule Explanation")

    st.write(
        "Storm is likely when wind speed is high and visibility is low."
    )