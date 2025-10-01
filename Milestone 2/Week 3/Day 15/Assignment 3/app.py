import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load("iris_model.pkl")
iris = load_iris()

st.title("🌸 Iris Flower Classifier")
st.write("Use the app to explore data and predict iris flower species.")

# Sidebar for mode selection
st.sidebar.title("📊 Navigation")
mode = st.sidebar.radio("Choose mode:", ["Prediction", "Exploration"])

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

if mode == "Prediction":
    st.subheader("Enter Flower Measurements")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, help="Length of sepal in cm")
        sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, help="Width of sepal in cm")
    with col2:
        petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0, help="Length of petal in cm")
        petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.2, help="Width of petal in cm")

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("Predict"):
        prediction = model.predict(features)[0]
        pred_proba = model.predict_proba(features)

        if prediction == 0:
            st.success(f"Prediction: {iris.target_names[prediction]}")
        elif prediction == 1:
            st.warning(f"Prediction: {iris.target_names[prediction]}")
        else:
            st.error(f"Prediction: {iris.target_names[prediction]}")

        st.write("Prediction probabilities:", pred_proba)

elif mode == "Exploration":
    st.subheader("Dataset Overview")
    st.write(df.head())

    plot_type = st.selectbox("Choose plot:", ["Histogram", "Scatterplot"])

    if plot_type == "Histogram":
        feature = st.selectbox("Select feature:", iris.feature_names)
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax)
        st.pyplot(fig)

    elif plot_type == "Scatterplot":
        x_feature = st.selectbox("X-axis:", iris.feature_names)
        y_feature = st.selectbox("Y-axis:", iris.feature_names)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_feature], y=df[y_feature], hue=df["species"], palette="deep", ax=ax)
        st.pyplot(fig)
