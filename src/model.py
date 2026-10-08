from pathlib import Path
import joblib
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age", "satisfaction_score", "properties_purchased", "total_investment",
    "average_property_price", "total_floor_area_sqft", "average_floor_area_sqft",
    "purchase_span_days", "has_property_purchase",
]
CATEGORICAL_FEATURES = [
    "client_type", "gender", "country", "region", "acquisition_purpose",
    "loan_applied", "referral_channel",
]


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def evaluate_kmeans(X, k_values=range(2, 9)):
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette_score": float(silhouette_score(X, labels)),
        })
    return pd.DataFrame(rows)


def fit_models(X, best_k: int):
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    k_labels = kmeans.fit_predict(X)
    hierarchical = AgglomerativeClustering(n_clusters=best_k)
    h_labels = hierarchical.fit_predict(X)
    return kmeans, hierarchical, k_labels, h_labels


def save_pipeline(preprocessor, kmeans, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": preprocessor, "kmeans": kmeans}, path)
