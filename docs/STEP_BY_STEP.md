# Step-by-Step Workflow
1. Place raw datasets in `data/raw/`.
2. Run `python run_pipeline.py`.
3. Inspect `outputs/reports/cluster_evaluation.csv`.
4. Inspect `outputs/reports/cluster_summary.csv`.
5. Inspect EDA, elbow and silhouette PNGs in `outputs/figures/`.
6. Launch `python -m streamlit run app.py`.
7. Use dashboard filters to compare countries, regions, purposes and client types.
