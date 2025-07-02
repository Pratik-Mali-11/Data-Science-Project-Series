from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_features(df, feature_cols):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    return pd.DataFrame(scaled, columns=feature_cols), scaler

def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    return pd.DataFrame(reduced, columns=[f"PC{i+1}" for i in range(n_components)]), pca

def perform_kmeans(data, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans
