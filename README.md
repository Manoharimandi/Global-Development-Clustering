# Global Development Cluster Analysis — Streamlit
## 🚀 Live Demo

[🌍 Open the Streamlit App](https://global-development-clustering-bteppwnxatgj5dqtjdicas.streamlit.app)

This package deploys the final K-Means model from the P693 Cluster Analysis project.

## Model
- K-Means, 2 clusters
- 208 countries
- Notebook-matching Silhouette Score: 0.309315
- Cluster 0: Lower Development Profile
- Cluster 1: Higher Development Profile

## Notebook-matching preprocessing
1. Convert currency/percentage-formatted columns to numeric.
2. Remove `Number of Records`.
3. Aggregate observations by country using the mean.
4. Median-impute missing values.
5. Apply `log1p` to features with skewness > 2.
6. Standardize with `StandardScaler`.
7. Predict using K-Means (`n_clusters=2`, `random_state=42`, `n_init=10`).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload all files in this folder to the repository root.
3. In Streamlit Community Cloud, create a new app.
4. Select the repository, branch, and `app.py`.
5. Deploy.
