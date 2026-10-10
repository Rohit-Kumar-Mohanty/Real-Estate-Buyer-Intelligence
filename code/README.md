# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

**Owner:** Rohit Kumar Mohanty  
**Domain:** Financial Analytics & Real Estate Market Intelligence

## Included
- Raw `clients.csv` and `properties.csv`
- Cleaning + client-property integration
- Buyer-level feature engineering
- One-hot encoding + StandardScaler
- K-Means clustering
- Hierarchical clustering
- Elbow and Silhouette evaluation
- EDA figures and cluster reports
- Persisted ML artifacts
- Streamlit dashboard with filters, owner name and live IST clock
- Research-paper template and viva questions

## Windows / VS Code setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_pipeline.py
python -m streamlit run app.py
```

If Matplotlib ever reports a Tkinter `init.tcl` error, run:
```powershell
$env:MPLBACKEND="Agg"
python run_pipeline.py
```

## Important
The PRD provides example segment concepts such as Global Investors, First-Time Buyers, Corporate Buyers and Luxury Investors. The clustering model does **not** hard-code these labels. Final segment names should be assigned only after inspecting the generated cluster summary.
