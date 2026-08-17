from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Cluster Analysis — Global Development",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling: clean dark dashboard
# -----------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.2rem;
            padding-left: 3.2rem;
            padding-right: 3.2rem;
            max-width: 1500px;
        }
        [data-testid="stSidebar"] {
            min-width: 310px;
            max-width: 310px;
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.12;
            margin-bottom: 0.5rem;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #b7bcc7;
            margin-bottom: 2.2rem;
        }
        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 0.7rem;
        }
        .status-card {
            padding: 0.9rem 1rem;
            border-radius: 10px;
            background: rgba(40, 167, 111, 0.20);
            border: 1px solid rgba(40, 167, 111, 0.35);
            color: #65e6a1;
            font-weight: 650;
        }
        .info-card {
            padding: 0.9rem 1rem;
            border-radius: 10px;
            background: rgba(40, 120, 190, 0.20);
            border: 1px solid rgba(70, 150, 220, 0.25);
        }
        div[data-testid="stMetric"] {
            padding: 0.2rem 0.4rem;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.82rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load project artifacts
# -----------------------------
@st.cache_resource
def load_metadata():
    metadata = joblib.load(BASE_DIR / "model_metadata.pkl")
    features = joblib.load(BASE_DIR / "feature_columns.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    model = joblib.load(BASE_DIR / "kmeans_model.pkl")
    return metadata, features, scaler, model


@st.cache_data
def load_project_data():
    return pd.read_csv(BASE_DIR / "country_development_profiles.csv")


metadata, saved_features, saved_scaler, saved_model = load_metadata()
project_data = load_project_data()

log_features = metadata.get("log_transformed_features", [])

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## ⚙️ Configuration")

uploaded = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"],
    help="Upload a country-level CSV containing Country and numeric development indicators.",
)

st.sidebar.caption("200MB per file • CSV")

if uploaded is not None:
    data = pd.read_csv(uploaded)
    source_name = uploaded.name
else:
    data = project_data.copy()
    source_name = "Built-in project dataset"

# Remove columns that are outputs rather than model inputs.
excluded = {
    "KMeans_Cluster",
    "KMeans_Profile",
    "Cluster",
    "cluster",
    "Profile",
}

country_col = "Country" if "Country" in data.columns else None

numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
feature_cols = [
    c for c in numeric_cols
    if c not in excluded and c in saved_features
]

# Fallback for a compatible uploaded country-level dataset.
if not feature_cols:
    feature_cols = [
        c for c in numeric_cols
        if c not in excluded
    ]

if not feature_cols or len(feature_cols) < 2:
    st.error(
        "The CSV does not contain enough numeric development features. "
        "Use the bundled project dataset or upload a compatible country-level CSV."
    )
    st.stop()

# Keep the project feature order whenever possible.
ordered = [c for c in saved_features if c in feature_cols]
remaining = [c for c in feature_cols if c not in ordered]
feature_cols = ordered + remaining

st.sidebar.markdown(
    f'<div class="status-card">Loaded: {len(data):,} countries × '
    f'{len(feature_cols)} features</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Clustering Algorithm")
algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["K-Means", "Agglomerative Clustering", "DBSCAN"],
    label_visibility="collapsed",
)

if algorithm in ["K-Means", "Agglomerative Clustering"]:
    n_clusters = st.sidebar.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=2,
        step=1,
    )
else:
    eps = st.sidebar.slider(
        "DBSCAN eps",
        min_value=0.5,
        max_value=5.0,
        value=2.75,
        step=0.05,
    )
    min_samples = st.sidebar.slider(
        "Minimum Samples",
        min_value=2,
        max_value=15,
        value=5,
        step=1,
    )

# -----------------------------
# Preprocessing
# -----------------------------
X = data[feature_cols].copy()
X = X.apply(pd.to_numeric, errors="coerce")

# For uploaded data, fill missing values using the uploaded data's medians.
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)

# Reproduce the project's log transformation for matching features.
for col in log_features:
    if col in X.columns:
        X[col] = np.log1p(np.maximum(X[col], 0))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Clustering
# -----------------------------
if algorithm == "K-Means":
    clusterer = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    labels = clusterer.fit_predict(X_scaled)

elif algorithm == "Agglomerative Clustering":
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="complete",
    )
    labels = clusterer.fit_predict(X_scaled)

else:
    clusterer = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )
    labels = clusterer.fit_predict(X_scaled)

labels = np.asarray(labels)
assigned_mask = labels != -1
assigned_X = X_scaled[assigned_mask]
assigned_labels = labels[assigned_mask]

unique_labels = np.unique(assigned_labels)

if len(unique_labels) >= 2 and len(assigned_X) > len(unique_labels):
    silhouette = silhouette_score(assigned_X, assigned_labels)
    davies = davies_bouldin_score(assigned_X, assigned_labels)
    calinski = calinski_harabasz_score(assigned_X, assigned_labels)
else:
    silhouette = np.nan
    davies = np.nan
    calinski = np.nan

coverage = float(assigned_mask.mean() * 100)
n_noise = int((labels == -1).sum())
n_clusters_found = len(unique_labels)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="hero-title">Cluster Analysis — Global<br>'
    'Development Measurements</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">'
    'Unsupervised ML project: Discover natural groupings of countries '
    'based on development indicators.'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# -----------------------------
# KPI row
# -----------------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("Algorithm", algorithm.replace(" Clustering", ""))
k2.metric(
    "Silhouette",
    "—" if np.isnan(silhouette) else f"{silhouette:.4f}",
)
k3.metric(
    "Davies-Bouldin",
    "—" if np.isnan(davies) else f"{davies:.4f}",
)
k4.metric(
    "Calinski-Harabasz",
    "—" if np.isnan(calinski) else f"{calinski:.2f}",
)

st.caption(
    f"Source: {source_name}  •  {len(data):,} countries  •  "
    f"{len(feature_cols)} numeric development features  •  "
    f"{coverage:.2f}% assigned"
)

# -----------------------------
# PCA projection
# -----------------------------
pca = PCA(n_components=2, random_state=42)
pca_values = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame(
    {
        "PC1": pca_values[:, 0],
        "PC2": pca_values[:, 1],
        "Cluster": labels.astype(str),
    }
)

if country_col:
    plot_df["Country"] = data[country_col].astype(str).values
    hover_name = "Country"
else:
    plot_df["Country"] = data.index.astype(str)
    hover_name = "Country"

st.markdown('<div class="section-title">PCA 2D Projection</div>', unsafe_allow_html=True)

fig = px.scatter(
    plot_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_name=hover_name,
    title=f"{algorithm} — PCA 2D Projection",
    template="plotly_dark",
)
fig.update_traces(marker={"size": 9, "line": {"width": 0.7}})
fig.update_layout(
    height=560,
    margin=dict(l=10, r=10, t=55, b=10),
    legend_title_text="Cluster",
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Project comparison
# -----------------------------
st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

comparison_rows = []

# K-Means final project configuration
km = KMeans(n_clusters=2, random_state=42, n_init=10)
km_labels = km.fit_predict(X_scaled)
comparison_rows.append(
    {
        "Model": "K-Means",
        "Clusters": 2,
        "Silhouette": silhouette_score(X_scaled, km_labels),
        "Coverage (%)": 100.0,
    }
)

# Hierarchical final comparison configuration
agg = AgglomerativeClustering(n_clusters=2, linkage="complete")
agg_labels = agg.fit_predict(X_scaled)
comparison_rows.append(
    {
        "Model": "Hierarchical — Complete",
        "Clusters": 2,
        "Silhouette": silhouette_score(X_scaled, agg_labels),
        "Coverage (%)": 100.0,
    }
)

# DBSCAN project configuration
db = DBSCAN(eps=2.75, min_samples=5)
db_labels = db.fit_predict(X_scaled)
db_mask = db_labels != -1
if len(np.unique(db_labels[db_mask])) >= 2:
    db_sil = silhouette_score(X_scaled[db_mask], db_labels[db_mask])
else:
    db_sil = np.nan

comparison_rows.append(
    {
        "Model": "DBSCAN",
        "Clusters": len(set(db_labels)) - (1 if -1 in db_labels else 0),
        "Silhouette": db_sil,
        "Coverage (%)": float(db_mask.mean() * 100),
    }
)

comparison_df = pd.DataFrame(comparison_rows)
comparison_df["Silhouette"] = comparison_df["Silhouette"].round(4)
comparison_df["Coverage (%)"] = comparison_df["Coverage (%)"].round(2)

st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# -----------------------------
# Cluster distribution
# -----------------------------
left, right = st.columns(2)

with left:
    st.markdown('<div class="section-title">Cluster Distribution</div>', unsafe_allow_html=True)
    counts = pd.Series(labels).value_counts().sort_index()
    dist = pd.DataFrame(
        {
            "Cluster": counts.index.astype(str),
            "Countries": counts.values,
        }
    )
    fig_dist = px.bar(
        dist,
        x="Cluster",
        y="Countries",
        title="Countries per Cluster",
        template="plotly_dark",
    )
    fig_dist.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_dist, use_container_width=True)

with right:
    st.markdown('<div class="section-title">Cluster Summary</div>', unsafe_allow_html=True)
    summary = dist.copy()
    summary["Share (%)"] = (
        summary["Countries"] / len(data) * 100
    ).round(2)
    if -1 in labels:
        summary.loc[summary["Cluster"] == "-1", "Cluster"] = "-1 (Noise)"
    st.dataframe(summary, use_container_width=True, hide_index=True)

# -----------------------------
# Country explorer
# -----------------------------
if country_col:
    st.markdown('<div class="section-title">Country Explorer</div>', unsafe_allow_html=True)

    selected_country = st.selectbox(
        "Select a country",
        sorted(data[country_col].astype(str).unique()),
    )

    selected_idx = data.index[data[country_col].astype(str) == selected_country][0]
    selected_position = data.index.get_loc(selected_idx)
    selected_cluster = int(labels[selected_position])

    c1, c2, c3 = st.columns(3)
    c1.metric("Country", selected_country)
    c2.metric("Cluster", selected_cluster)
    c3.metric(
        "Profile",
        "Noise / Outlier"
        if selected_cluster == -1
        else (
            "Lower Development Profile"
            if selected_cluster == 0 and algorithm == "K-Means" and n_clusters == 2
            else "Clustered Development Profile"
        ),
    )

    country_values = data.loc[selected_idx, feature_cols].to_frame("Value")
    country_values.index.name = "Indicator"
    st.dataframe(country_values, use_container_width=True)

# -----------------------------
# Interpretation
# -----------------------------
st.markdown('<div class="section-title">How to read the metrics</div>', unsafe_allow_html=True)
st.markdown(
    """
    - **Silhouette:** higher is generally better; it measures how well separated the clusters are.
    - **Davies-Bouldin:** lower is generally better; it compares within-cluster compactness with separation.
    - **Calinski-Harabasz:** higher is generally better; it rewards compact and well-separated clusters.
    - **Coverage:** percentage of countries assigned to a cluster. DBSCAN can mark some countries as noise (`-1`).
    """
)

st.divider()
st.caption(
    "P693 Cluster Analysis • Global Development Measurement Dataset • "
    "K-Means final deployment model"
)
