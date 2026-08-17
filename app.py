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

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Cluster Analysis — Global Development Measurements",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Styling: intentionally close to the reference dashboard layout
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.0rem;
            padding-left: 3.2rem;
            padding-right: 3.2rem;
            max-width: 1500px;
        }

        [data-testid="stSidebar"] {
            min-width: 310px;
            max-width: 310px;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 2.0rem;
        }

        .reference-title {
            font-size: 3.0rem;
            line-height: 1.10;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin: 0 0 0.55rem 0;
        }

        .reference-subtitle {
            font-size: 1.0rem;
            font-weight: 600;
            color: #d1d5db;
            margin-bottom: 2.7rem;
        }

        .sidebar-heading {
            font-size: 1.25rem;
            font-weight: 750;
            margin-bottom: 1.0rem;
        }

        .loaded-card {
            background: rgba(52, 168, 112, 0.23);
            border: 1px solid rgba(72, 190, 130, 0.30);
            border-radius: 10px;
            padding: 0.9rem 0.9rem;
            color: #58e69b;
            font-weight: 700;
            line-height: 1.35;
            margin: 0.8rem 0 1.2rem 0;
        }

        .sidebar-caption {
            color: #9ca3af;
            font-size: 0.78rem;
            margin-top: 0.4rem;
            margin-bottom: 1.25rem;
        }

        .metric-label {
            color: #c7cbd4;
            font-size: 0.86rem;
            font-weight: 650;
        }

        .metric-value {
            font-size: 1.75rem;
            font-weight: 500;
            margin-top: 0.25rem;
        }

        div[data-testid="stMetric"] {
            padding: 0;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.86rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.75rem;
            font-weight: 500;
        }

        .small-section {
            font-size: 1.15rem;
            font-weight: 750;
            margin: 1.5rem 0 0.65rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Project artifacts
# -------------------------------------------------------------------
@st.cache_resource
def load_project_artifacts():
    metadata = joblib.load(BASE_DIR / "model_metadata.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    final_model = joblib.load(BASE_DIR / "kmeans_model.pkl")
    return metadata, feature_columns, scaler, final_model


@st.cache_data
def load_project_dataset():
    return pd.read_csv(BASE_DIR / "country_development_profiles.csv")


metadata, saved_features, saved_scaler, saved_model = load_project_artifacts()
default_data = load_project_dataset()
log_features = metadata.get("log_transformed_features", [])

# -------------------------------------------------------------------
# Sidebar — reference-style configuration panel
# -------------------------------------------------------------------
st.sidebar.markdown(
    '<div class="sidebar-heading">⚙️ Configuration</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"],
    help="Upload a compatible country-level development dataset.",
)

st.sidebar.markdown(
    '<div class="sidebar-caption">200MB per file • CSV</div>',
    unsafe_allow_html=True,
)

if uploaded_file is None:
    data = default_data.copy()
    source_label = "Built-in project dataset"
else:
    data = pd.read_csv(uploaded_file)
    source_label = uploaded_file.name

# The project's model input columns are exactly the 23 saved features.
available_features = [c for c in saved_features if c in data.columns]

if len(available_features) != len(saved_features):
    missing = [c for c in saved_features if c not in data.columns]
    st.error(
        "The uploaded CSV is not compatible with this project. "
        f"Missing required feature columns: {', '.join(missing)}"
    )
    st.stop()

feature_columns = saved_features.copy()

st.sidebar.markdown(
    f'<div class="loaded-card">Loaded: {len(data):,} countries × '
    f'{len(feature_columns)}<br>features</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Clustering Algorithm")

algorithm = st.sidebar.selectbox(
    "Clustering Algorithm",
    ["K-Means", "Hierarchical", "DBSCAN"],
    index=0,
    label_visibility="collapsed",
)

if algorithm in ["K-Means", "Hierarchical"]:
    st.sidebar.markdown("### Number of Clusters")
    n_clusters = st.sidebar.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=2,
        step=1,
        label_visibility="collapsed",
    )
else:
    st.sidebar.markdown("### DBSCAN Parameters")
    eps = st.sidebar.slider(
        "eps",
        min_value=0.50,
        max_value=5.00,
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

# -------------------------------------------------------------------
# Exact project preprocessing
# -------------------------------------------------------------------
X = data[feature_columns].copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# For the bundled project dataset, use the same saved scaler used by
# the final deployed K-Means model. For uploaded compatible data, use
# the same scaler so the input space remains consistent.
X = X.fillna(X.median(numeric_only=True)).fillna(0)

for col in log_features:
    if col in X.columns:
        X[col] = np.log1p(np.maximum(X[col], 0))

X_scaled = saved_scaler.transform(X)

# -------------------------------------------------------------------
# Clustering
# -------------------------------------------------------------------
if algorithm == "K-Means":
    if uploaded_file is None and n_clusters == 2:
        # Use the exact saved final model for the default project view.
        labels = saved_model.predict(X_scaled)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

elif algorithm == "Hierarchical":
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="complete",
    )
    labels = model.fit_predict(X_scaled)

else:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

labels = np.asarray(labels)

assigned = labels != -1
assigned_X = X_scaled[assigned]
assigned_labels = labels[assigned]

if len(np.unique(assigned_labels)) >= 2:
    silhouette = silhouette_score(assigned_X, assigned_labels)
    davies = davies_bouldin_score(assigned_X, assigned_labels)
    calinski = calinski_harabasz_score(assigned_X, assigned_labels)
else:
    silhouette = np.nan
    davies = np.nan
    calinski = np.nan

coverage = float(assigned.mean() * 100)
noise_count = int((labels == -1).sum())
cluster_count = len(np.unique(assigned_labels))

# -------------------------------------------------------------------
# Main header
# -------------------------------------------------------------------
st.markdown(
    '<div class="reference-title">Cluster Analysis — Global<br>'
    'Development Measurements</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="reference-subtitle">'
    'Unsupervised ML project: Discover natural groupings of countries '
    'based on development indicators.'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# -------------------------------------------------------------------
# Reference-style metric row
# -------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

m1.metric("Algorithm", algorithm)
m2.metric(
    "Silhouette",
    "—" if np.isnan(silhouette) else f"{silhouette:.4f}",
)
m3.metric(
    "Davies-Bouldin",
    "—" if np.isnan(davies) else f"{davies:.4f}",
)
m4.metric(
    "Calinski-Harabasz",
    "—" if np.isnan(calinski) else f"{calinski:.2f}",
)

# -------------------------------------------------------------------
# PCA 2D visualization — directly below metrics like reference
# -------------------------------------------------------------------
st.markdown('<div class="small-section">PCA 2D Projection</div>', unsafe_allow_html=True)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame(
    {
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Cluster": labels.astype(str),
    }
)

if "Country" in data.columns:
    plot_df["Country"] = data["Country"].astype(str).values
else:
    plot_df["Country"] = data.index.astype(str)

fig = px.scatter(
    plot_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_name="Country",
    title=f"{algorithm} — PCA 2D Projection",
    template="plotly_dark",
)

fig.update_traces(
    marker=dict(size=8, line=dict(width=0.7)),
)

fig.update_layout(
    height=610,
    margin=dict(l=0, r=0, t=55, b=0),
    legend_title_text="Cluster",
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displaylogo": False,
        "responsive": True,
    },
)

# -------------------------------------------------------------------
# Below-the-fold analysis
# -------------------------------------------------------------------
st.markdown('<div class="small-section">Cluster Distribution</div>', unsafe_allow_html=True)

counts = pd.Series(labels).value_counts().sort_index()
distribution = pd.DataFrame(
    {
        "Cluster": counts.index.astype(str),
        "Countries": counts.values,
        "Share (%)": (counts.values / len(data) * 100).round(2),
    }
)

d1, d2 = st.columns([1.4, 1])

with d1:
    dist_fig = px.bar(
        distribution,
        x="Cluster",
        y="Countries",
        template="plotly_dark",
        title="Countries per Cluster",
    )
    dist_fig.update_layout(height=360, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(dist_fig, use_container_width=True)

with d2:
    st.dataframe(distribution, use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# Fixed project model comparison
# -------------------------------------------------------------------
st.markdown('<div class="small-section">Model Comparison</div>', unsafe_allow_html=True)

# These are recomputed from the same standardized project feature space,
# using the final configurations used in the project.
km_labels = saved_model.predict(X_scaled)
km_sil = silhouette_score(X_scaled, km_labels)
km_db = davies_bouldin_score(X_scaled, km_labels)
km_ch = calinski_harabasz_score(X_scaled, km_labels)

hc_labels = AgglomerativeClustering(
    n_clusters=2,
    linkage="complete",
).fit_predict(X_scaled)
hc_sil = silhouette_score(X_scaled, hc_labels)
hc_db = davies_bouldin_score(X_scaled, hc_labels)
hc_ch = calinski_harabasz_score(X_scaled, hc_labels)

db_labels = DBSCAN(
    eps=2.75,
    min_samples=5,
).fit_predict(X_scaled)
db_mask = db_labels != -1

if len(np.unique(db_labels[db_mask])) >= 2:
    db_sil = silhouette_score(X_scaled[db_mask], db_labels[db_mask])
    db_db = davies_bouldin_score(X_scaled[db_mask], db_labels[db_mask])
    db_ch = calinski_harabasz_score(X_scaled[db_mask], db_labels[db_mask])
else:
    db_sil = db_db = db_ch = np.nan

comparison = pd.DataFrame(
    [
        {
            "Model": "K-Means",
            "Clusters": 2,
            "Silhouette": km_sil,
            "Davies-Bouldin": km_db,
            "Calinski-Harabasz": km_ch,
            "Coverage (%)": 100.0,
        },
        {
            "Model": "Hierarchical — Complete",
            "Clusters": 2,
            "Silhouette": hc_sil,
            "Davies-Bouldin": hc_db,
            "Calinski-Harabasz": hc_ch,
            "Coverage (%)": 100.0,
        },
        {
            "Model": "DBSCAN",
            "Clusters": len(set(db_labels)) - (1 if -1 in db_labels else 0),
            "Silhouette": db_sil,
            "Davies-Bouldin": db_db,
            "Calinski-Harabasz": db_ch,
            "Coverage (%)": db_mask.mean() * 100,
        },
    ]
)

for col in ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Coverage (%)"]:
    comparison[col] = comparison[col].round(4 if col != "Coverage (%)" else 2)

st.dataframe(comparison, use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# Country explorer
# -------------------------------------------------------------------
if "Country" in data.columns:
    st.markdown('<div class="small-section">Country Explorer</div>', unsafe_allow_html=True)

    selected_country = st.selectbox(
        "Select a country",
        sorted(data["Country"].astype(str).unique()),
    )

    row_pos = data.index[data["Country"].astype(str) == selected_country][0]
    pos = data.index.get_loc(row_pos)
    selected_cluster = int(labels[pos])

    c1, c2, c3 = st.columns(3)
    c1.metric("Country", selected_country)
    c2.metric("Cluster", selected_cluster)
    if selected_cluster == -1:
        profile = "Noise / Outlier"
    elif algorithm == "K-Means" and n_clusters == 2:
        profile = (
            "Lower Development Profile"
            if selected_cluster == 0
            else "Higher Development Profile"
        )
    else:
        profile = "Clustered Development Profile"
    c3.metric("Development Profile", profile)

    country_values = data.loc[row_pos, feature_columns].to_frame("Value")
    country_values.index.name = "Indicator"
    st.dataframe(country_values, use_container_width=True)

st.divider()
st.caption(
    f"Global Development Measurements • {len(data):,} countries × "
    f"{len(feature_columns)} features • {coverage:.2f}% assigned • "
    f"Source: {source_label}"
)
