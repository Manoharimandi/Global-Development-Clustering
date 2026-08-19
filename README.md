🌍 Cluster Analysis --- Global Development Measurements

An interactive Unsupervised Machine Learning application for
discovering natural groups in country-level development data.

🚀 Live Demo

Streamlit App:
https://global-development-clustering-bteppwnxatgj5dqtjdicas.streamlit.app

🎯 Project Objective

The goal is to group countries or other observations based on multiple
numerical development indicators such as GDP, Birth Rate, CO2 Emissions,
Health Expenditure, Energy Usage, Internet Usage, Life Expectancy and
other economic and social measurements.

Because there is no predefined target label, this is an unsupervised
learning problem.

🧠 Machine Learning Workflow

CSV Upload
   ↓
Data Validation
   ↓
Numeric Feature Selection
   ↓
Missing / Invalid Value Handling
   ↓
StandardScaler
   ↓
Clustering
   ↓
Evaluation Metrics
   ↓
PCA 2D Visualization
   ↓
Cluster Distribution
   ↓
Model Comparison
   ↓
Country Explorer
   ↓
Download Results CSV

🤖 Clustering Algorithms

K-Means

Groups observations around cluster centroids. The user can select the
number of clusters from the sidebar.

Hierarchical Agglomerative Clustering

Uses agglomerative clustering with complete linkage, progressively
merging observations into groups.

DBSCAN

A density-based clustering algorithm that can identify dense groups and
noise/outlier observations. The application exposes eps and
min_samples.

📊 Cluster Evaluation

Metric                  Meaning                 Better

Silhouette Score        Measures how well       Higher
observations fit their
own cluster compared
with other clusters

Davies-Bouldin Index    Measures similarity     Lower
between clusters

📉 PCA Visualization

The application uses Principal Component Analysis (PCA) to project
the standardized multidimensional feature space into two dimensions, PC1
and PC2, for interactive visualization with Plotly.

PCA is used primarily for visualization; clustering is performed on the
standardized numerical feature space.

📁 Custom CSV Upload

The application is upload-first. It does not require visitors to use
the built-in project dataset.

After a CSV is uploaded, the application:

Reads it with Pandas.

Validates that it contains data.

Detects numerical columns automatically.

Requires at least two numerical features.

Detects common country-column names such as Country, country,
Country Name, and country_name.

Converts invalid numeric values to missing values.

Fills missing numeric values with column medians.

Standardizes the numerical features.

Runs the selected clustering algorithm.

The country identifier is not used as a clustering feature.

📈 Cluster Distribution

The dashboard displays the number and percentage of records in each
cluster using an interactive chart and table.

🔬 Model Comparison

The application compares K-Means, Hierarchical --- Complete Linkage, and
DBSCAN using the same uploaded dataset.

Benchmark settings used in the comparison are:

K-Means: 2 clusters

Hierarchical: 2 clusters, complete linkage

DBSCAN: eps=2.75, min_samples=5

The main sidebar controls can use different settings.

🌎 Country Explorer

When a recognized country column is available, users can select a
country and view its assigned cluster and numerical development
indicators.

📥 Download Results

Users can download the processed dataset as:

clustering_results.csv

The downloaded file contains all original columns plus a Cluster
column containing the current clustering assignment.

🛠️ Technology Stack

Python

Pandas

NumPy

Scikit-learn

Plotly

Streamlit

GitHub

Streamlit Community Cloud

📂 Project Structure

Global-Development-Clustering/
│
├── app.py
├── README.md
├── requirements.txt
├── country_development_profiles.csv
├── feature_columns.pkl
├── kmeans_model.pkl
├── model_metadata.pkl
├── scaler.pkl
└── .devcontainer/

Important files

app.py --- Complete Streamlit application, preprocessing,
clustering, evaluation, visualizations, Country Explorer and CSV
download.

requirements.txt --- Python dependencies required by the
application.

country_development_profiles.csv --- Original project
country-development dataset.

.pkl files --- Serialized project/model artifacts retained in the
repository.

▶️ Run Locally

git clone https://github.com/Manoharimandi/Global-Development-Clustering.git
cd Global-Development-Clustering
pip install -r requirements.txt
streamlit run app.py

Then open the local Streamlit URL, normally:

http://localhost:8501

☁️ Deployment

The application is deployed through Streamlit Community Cloud using the
GitHub repository.

Python Application
      ↓
GitHub Repository
      ↓
Streamlit Community Cloud
      ↓
Public Web Application

⚠️ Input Requirements

For meaningful clustering results, an uploaded CSV should contain at
least two numerical features. A Country column is recommended if
the Country Explorer is required.

📌 Key Features

✅ Custom CSV upload

✅ Automatic numeric feature detection

✅ Missing-value handling

✅ Feature standardization

✅ K-Means

✅ Hierarchical Agglomerative Clustering

✅ DBSCAN

✅ Silhouette Score

✅ Davies-Bouldin Index

✅ Calinski-Harabasz Score

✅ PCA 2D visualization

✅ Interactive Plotly charts

✅ Cluster distribution

✅ Model comparison

✅ Country Explorer

✅ Download clustered CSV

✅ Streamlit deployment

📚 Project Summary

This project demonstrates how unsupervised machine learning can
discover hidden patterns and natural groupings in global development
data. The Streamlit application converts the complete clustering
workflow into an interactive tool where users can upload their own
dataset, select algorithms and parameters, evaluate cluster quality,
visualize the results, explore country-level assignments, and download
the clustered dataset.

