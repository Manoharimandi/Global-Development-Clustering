from pathlib import Path

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
    page_title="Cluster Analysis — Global Development Measurements",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Reference-style visual design
# -----------------------------
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
            padding-top: 2rem;
        }

        .main-title {
            font-size: 3.0rem;
            line-height: 1.08;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin: 0 0 0.55rem 0;
        }

        .subtitle {
            font-size: 1rem;
            font-weight: 600;
            color: #d1d5db;
            margin-bottom: 2.7rem;
        }

        .sidebar-title {
            font-size: 1.25rem;
            font-weight: 750;
            margin-bottom: 1rem;
        }

        .loaded-card {
            background: rgba(52, 168, 112, 0.23);
            border: 1px solid rgba(72, 190, 130, 0.30);
            border-radius: 10px;
            padding: 0.9rem;
            color: #58e69b;
            font-weight: 700;
            line-height: 1.35;
            margin: 0.8rem 0 1.2rem;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 750;
            margin: 1.5rem 0 0.65rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">Cluster Analysis — Global<br>'
    'Development Measurements</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Unsupervised ML project: Discover natural groupings '
    'of countries based on development indicators.</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown(
    '<div class="sidebar-title">⚙️ Configuration</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"],
    help="Upload your country-level development measurements CSV.",
)

st.sidebar.caption("200MB per file • CSV")

# -----------------------------
# FIRST SCREEN: no built-in data
# -----------------------------
if uploaded_file is None:
    st.divider()

    st.info("👉 Upload a CSV file from the sidebar to get started.")

    st.markdown(
        "**Expected format:** World Development Measurements CSV with columns "
        "like `Birth Rate`, `GDP`, `Country`, etc."
    )

    st.stop()

# -----------------------------
# Read user's dataset
# -----------------------------
try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV is empty. Please upload a CSV containing data.")
    st.stop()

# Country column is preferred but not mandatory.
country_col = None
for candidate in ["Country", "country", "Country Name", "country_name"]:
    if candidate in data.columns:
        country_col = candidate
        break

# Use numeric columns for clustering.
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error(
        "The uploaded dataset needs at least 2 numeric development-indicator "
        "columns for clustering."
    )
    st.stop()

# Keep the project country identifier out of the numeric feature matrix.
if country_col in numeric_cols:
    numeric_cols.remove(country_col)

if len(numeric_cols) < 2:
    st.error("At least 2 numeric feature columns are required.")
    st.stop()

# -----------------------------
# Sidebar controls after upload
# -----------------------------
st.sidebar.markdown(
    f'<div class="loaded-card">Loaded: {len(data):,} rows × '
    f'{len(numeric_cols)}<br>numeric features</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Clustering Algorithm")

algorithm = st.sidebar.selectbox(
    "Clustering Algorithm",
    ["K-Means", "Hierarchical", "DBSCAN"],
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
        min_value=0.10,
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

# -----------------------------
# Preprocessing
# -----------------------------
X = data[numeric_cols].copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing numeric values with column medians.
X = X.fillna(X.median(numeric_only=True)).fillna(0)

# Standardize the uploaded dataset itself.
# This makes the application independent of the built-in project CSV.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Clustering
# -----------------------------
if algorithm == "K-Means":
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    labels = model.fit_predict(X_scaled)

elif algorithm == "Hierarchical":
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="complete",
    )
    labels = model.fit_predict(X_scaled)

else:
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )
    labels = model.fit_predict(X_scaled)

labels = np.asarray(labels)

# -----------------------------
# Evaluation
# -----------------------------
assigned = labels != -1
assigned_X = X_scaled[assigned]
assigned_labels = labels[assigned]

if len(np.unique(assigned_labels)) >= 2 and len(assigned_labels) > 2:
    silhouette = silhouette_score(assigned_X, assigned_labels)
    davies = davies_bouldin_score(assigned_X, assigned_labels)
    calinski = calinski_harabasz_score(assigned_X, assigned_labels)
else:
    silhouette = np.nan
    davies = np.nan
    calinski = np.nan

coverage = assigned.mean() * 100

# -----------------------------
# Metric row
# -----------------------------
st.divider()

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

# -----------------------------
# PCA visualization
# -----------------------------
st.markdown(
    '<div class="section-title">PCA 2D Projection</div>',
    unsafe_allow_html=True,
)

pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame(
    {
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Cluster": labels.astype(str),
    }
)

if country_col:
    plot_df["Country"] = data[country_col].astype(str).values
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
    marker=dict(size=8, line=dict(width=0.7))
)
fig.update_layout(
    height=610,
    margin=dict(l=0, r=0, t=55, b=0),
    legend_title_text="Cluster",
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displaylogo": False, "responsive": True},
)

# -----------------------------
# Cluster distribution
# -----------------------------
st.markdown(
    '<div class="section-title">Cluster Distribution</div>',
    unsafe_allow_html=True,
)

counts = pd.Series(labels).value_counts().sort_index()

distribution = pd.DataFrame(
    {
        "Cluster": counts.index.astype(str),
        "Countries / Rows": counts.values,
        "Share (%)": (counts.values / len(data) * 100).round(2),
    }
)

d1, d2 = st.columns([1.4, 1])

with d1:
    dist_fig = px.bar(
        distribution,
        x="Cluster",
        y="Countries / Rows",
        title="Records per Cluster",
        template="plotly_dark",
    )
    dist_fig.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(dist_fig, use_container_width=True)

with d2:
    st.dataframe(
        distribution,
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------
# Model comparison on the uploaded dataset
# -----------------------------
st.markdown(
    '<div class="section-title">Model Comparison</div>',
    unsafe_allow_html=True,
)

def evaluate_labels(x, y):
    mask = y != -1
    xx = x[mask]
    yy = y[mask]
    if len(np.unique(yy)) < 2 or len(yy) <= 2:
        return np.nan, np.nan, np.nan, mask.mean() * 100
    return (
        silhouette_score(xx, yy),
        davies_bouldin_score(xx, yy),
        calinski_harabasz_score(xx, yy),
        mask.mean() * 100,
    )

comparison_rows = []

km = KMeans(n_clusters=2, random_state=42, n_init=10)
km_y = km.fit_predict(X_scaled)
s, db, ch, cov = evaluate_labels(X_scaled, km_y)
comparison_rows.append(["K-Means", 2, s, db, ch, cov])

hc = AgglomerativeClustering(n_clusters=2, linkage="complete")
hc_y = hc.fit_predict(X_scaled)
s, db, ch, cov = evaluate_labels(X_scaled, hc_y)
comparison_rows.append(["Hierarchical — Complete", 2, s, db, ch, cov])

dbscan = DBSCAN(eps=2.75, min_samples=5)
db_y = dbscan.fit_predict(X_scaled)
db_count = len(set(db_y)) - (1 if -1 in db_y else 0)
s, db, ch, cov = evaluate_labels(X_scaled, db_y)
comparison_rows.append(["DBSCAN", db_count, s, db, ch, cov])

comparison = pd.DataFrame(
    comparison_rows,
    columns=[
        "Model",
        "Clusters",
        "Silhouette",
        "Davies-Bouldin",
        "Calinski-Harabasz",
        "Coverage (%)",
    ],
)

st.dataframe(
    comparison.round(
        {
            "Silhouette": 4,
            "Davies-Bouldin": 4,
            "Calinski-Harabasz": 2,
            "Coverage (%)": 2,
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------
# Country explorer when Country exists
# -----------------------------
if country_col:
    st.markdown(
        '<div class="section-title">Country Explorer</div>',
        unsafe_allow_html=True,
    )

    countries = sorted(data[country_col].astype(str).unique())
    selected_country = st.selectbox("Select a country", countries)

    selected_positions = np.where(
        data[country_col].astype(str).values == selected_country
    )[0]
    pos = int(selected_positions[0])
    selected_cluster = int(labels[pos])

    c1, c2, c3 = st.columns(3)
    c1.metric("Country", selected_country)
    c2.metric("Cluster", selected_cluster)
    c3.metric(
        "Development Profile",
        "Noise / Outlier" if selected_cluster == -1 else "Clustered Profile",
    )

    country_values = data.iloc[pos][numeric_cols].to_frame("Value")
    country_values.index.name = "Indicator"
    st.dataframe(country_values, use_container_width=True)

st.divider()
st.caption(
    f"Uploaded dataset • {len(data):,} rows × {len(numeric_cols)} numeric "
    f"features • {coverage:.2f}% assigned"
)
