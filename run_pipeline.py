from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_pipeline import clean_and_integrate
from src.eda import save_eda_figures
from src.model import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_preprocessor,
    evaluate_kmeans, fit_models, save_pipeline,
)

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ROOT / "models"
for p in [PROCESSED, REPORTS, FIGURES, MODELS]:
    p.mkdir(parents=True, exist_ok=True)

print("[1/7] Loading, cleaning and integrating datasets...")
df = clean_and_integrate(RAW / "clients.csv", RAW / "properties.csv")
df.to_csv(PROCESSED / "buyer_analytical_dataset.csv", index=False)
print(f"    Buyer records: {len(df):,}")

print("[2/7] Creating EDA figures...")
save_eda_figures(df, FIGURES)

print("[3/7] Encoding and scaling features...")
preprocessor = build_preprocessor()
X = preprocessor.fit_transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])

print("[4/7] Evaluating K-Means for k=2..8...")
evaluation = evaluate_kmeans(X, range(2, 9))
evaluation.to_csv(REPORTS / "cluster_evaluation.csv", index=False)
best_k = int(evaluation.loc[evaluation["silhouette_score"].idxmax(), "k"])
print(f"    Best k by silhouette score: {best_k}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(evaluation["k"], evaluation["inertia"], marker="o")
ax.set_title("Elbow Method")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Inertia")
fig.tight_layout(); fig.savefig(FIGURES / "elbow_method.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(evaluation["k"], evaluation["silhouette_score"], marker="o")
ax.set_title("Silhouette Score by k")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Silhouette Score")
fig.tight_layout(); fig.savefig(FIGURES / "silhouette_scores.png", dpi=160); plt.close(fig)

print("[5/7] Fitting K-Means and hierarchical clustering...")
kmeans, hierarchical, k_labels, h_labels = fit_models(X, best_k)
out = df.copy()
out["kmeans_cluster"] = k_labels
out["hierarchical_cluster"] = h_labels
out.to_csv(PROCESSED / "buyer_clustered_dataset.csv", index=False)

summary = out.groupby("kmeans_cluster").agg(
    buyers=("client_id", "count"),
    avg_age=("age", "mean"),
    avg_satisfaction=("satisfaction_score", "mean"),
    avg_properties_purchased=("properties_purchased", "mean"),
    avg_total_investment=("total_investment", "mean"),
    total_investment=("total_investment", "sum"),
    avg_property_price=("average_property_price", "mean"),
    loan_users=("loan_applied", lambda s: (s.astype(str).str.lower() == "yes").sum()),
    investment_purpose_buyers=("acquisition_purpose", lambda s: (s.astype(str).str.lower() == "investment").sum()),
).reset_index()
summary["investment_purpose_pct"] = summary["investment_purpose_buyers"] / summary["buyers"] * 100
summary["loan_user_pct"] = summary["loan_users"] / summary["buyers"] * 100
summary.to_csv(REPORTS / "cluster_summary.csv", index=False)

print("[6/7] Saving trained artifacts...")
save_pipeline(preprocessor, kmeans, MODELS / "buyer_segmentation_pipeline.joblib")
import joblib
joblib.dump(hierarchical, MODELS / "hierarchical_clustering.joblib")

with open(REPORTS / "pipeline_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Best K: {best_k}\n")
    f.write(f"Buyer records: {len(out)}\n")
    f.write(f"Sold purchases linked to clients: {int(out['properties_purchased'].sum())}\n")
    f.write("Features used:\n")
    for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        f.write(f"- {c}\n")

print("[7/7] Pipeline complete.")
print(f"    Processed dataset: {PROCESSED / 'buyer_clustered_dataset.csv'}")
print(f"    Cluster summary:   {REPORTS / 'cluster_summary.csv'}")
print(f"    Model artifact:    {MODELS / 'buyer_segmentation_pipeline.joblib'}")
