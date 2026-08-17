from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "kmeans_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    features = joblib.load(BASE_DIR / "feature_columns.pkl")
    metadata = joblib.load(BASE_DIR / "model_metadata.pkl")
    return model, scaler, features, metadata

@st.cache_data
def load_country_data():
    return pd.read_csv(BASE_DIR / "country_development_profiles.csv")

st.set_page_config(
    page_title="Global Development Cluster Analysis",
    page_icon="🌍",
    layout="wide",
)

model, scaler, feature_cols, metadata = load_artifacts()
country_data = load_country_data()
cluster_names = metadata["cluster_names"]
log_features = metadata["log_transformed_features"]

st.title("🌍 Global Development Cluster Analysis")
st.caption("K-Means clustering of country-level global development indicators")

page = st.sidebar.radio(
    "Choose a section",
    ["Country Explorer", "Custom Profile Prediction", "Model Overview"],
)

if page == "Country Explorer":
    st.header("Country Explorer")
    country = st.selectbox("Select a country", sorted(country_data["Country"].unique()))
    row = country_data[country_data["Country"] == country].iloc[0]

    cluster = int(row["KMeans_Cluster"])
    profile = row["KMeans_Profile"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Country", country)
    c2.metric("Cluster", cluster)
    c3.metric("Development Profile", profile)

    st.subheader("Development Indicators")
    indicators = row[feature_cols].to_frame("Value")
    indicators.index.name = "Indicator"
    st.dataframe(indicators, use_container_width=True)

    st.subheader("Cluster Distribution")
    distribution = (
        country_data["KMeans_Profile"]
        .value_counts()
        .rename_axis("Profile")
        .reset_index(name="Countries")
    )
    distribution["Percentage"] = (
        distribution["Countries"] / len(country_data) * 100
    ).round(2)
    st.bar_chart(distribution.set_index("Profile")["Countries"])
    st.dataframe(distribution, use_container_width=True)

elif page == "Custom Profile Prediction":
    st.header("Predict a Custom Country Profile")
    st.write(
        "Enter a country-level profile using the same units as the source dataset. "
        "The app applies the notebook's log1p transformation and StandardScaler "
        "before making the K-Means prediction."
    )
    st.info(
        "Example: GDP is entered as a numeric amount after removing $ and commas. "
        "The default values are the training-data medians."
    )

    default_row = country_data[feature_cols].median()
    values = {}
    cols = st.columns(2)

    for i, feature in enumerate(feature_cols):
        with cols[i % 2]:
            values[feature] = st.number_input(
                feature,
                value=float(default_row[feature]),
                format="%.6g",
                key=f"input_{feature}",
            )

    if st.button("Predict Cluster", type="primary"):
        input_df = pd.DataFrame([values], columns=feature_cols)

        for feature in log_features:
            input_df[feature] = np.log1p(np.maximum(input_df[feature], 0))

        prediction = int(model.predict(scaler.transform(input_df))[0])
        profile = cluster_names[prediction]

        st.success(f"Predicted: Cluster {prediction} — {profile}")

elif page == "Model Overview":
    st.header("Model Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Final Model", metadata["model"])
    c2.metric("Clusters", metadata["n_clusters"])
    c3.metric("Silhouette Score", f'{metadata["silhouette_score"]:.4f}')

    st.subheader("Comparative Analysis")
    comparison = pd.DataFrame({
        "Model": [
            "K-Means",
            "Hierarchical - Complete Linkage",
            "DBSCAN",
        ],
        "Clusters": [2, 2, 3],
        "Silhouette Score": [0.309315, 0.2943, 0.2660],
        "Coverage (%)": [100.00, 100.00, 73.08],
    })
    st.dataframe(comparison, use_container_width=True)

    st.subheader("Final Model Selection")
    st.write(
        "K-Means with two clusters is the selected deployment model. "
        "The project describes the resulting segments as Lower Development "
        "Profile and Higher Development Profile."
    )

    st.subheader("Features Used")
    st.write(", ".join(feature_cols))

    st.subheader("Log-Transformed Features")
    st.write(", ".join(log_features))

st.divider()
st.caption("Project: Cluster Analysis — Global Development Measurement Dataset")
