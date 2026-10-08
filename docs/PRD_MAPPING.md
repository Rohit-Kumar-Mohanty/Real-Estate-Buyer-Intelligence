# PRD Mapping
- Data cleaning: implemented in `src/data_pipeline.py`.
- Feature encoding: OneHotEncoder for categorical buyer attributes.
- Feature scaling: StandardScaler for numeric features.
- Client-property integration: sold properties are aggregated to buyer level.
- Clustering: K-Means and Agglomerative/Hierarchical clustering.
- Optimal k: silhouette score across k=2..8, with elbow plot generated.
- Interpretation: `cluster_summary.csv` and dashboard insights.
- Dashboard: segmentation, investor behavior, geography, insights and filters.
