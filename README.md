# 🌍 Global Development Cluster Analysis

> Unsupervised machine learning project for discovering country groupings from global development indicators.

## 🚀 Live Demo

https://global-development-clustering-bteppwnxatgj5dqtjdicas.streamlit.app

## What the application does

- Loads the processed 208-country development dataset automatically.
- Supports K-Means, Agglomerative/Hierarchical Clustering, and DBSCAN.
- Calculates Silhouette, Davies-Bouldin, Calinski-Harabasz and coverage metrics.
- Visualizes clusters using a 2D PCA projection.
- Shows model comparison and cluster distribution.
- Provides a country explorer.
- Supports optional CSV upload for compatible country-level datasets.

## Final project model

The project selected **K-Means with 2 clusters** as the deployment model, with a Silhouette Score of approximately **0.3093** and 100% country coverage.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

