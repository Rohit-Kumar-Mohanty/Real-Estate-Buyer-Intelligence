from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
# Streamlit may run this file from a nested folder (the deployed path shown was
# .../real-estate-buyer-intelligence/code). Search the app folder AND its parent
# repository folders, because data/ may live beside code/ rather than inside it.
SEARCH_ROOTS = [ROOT, *list(ROOT.parents)[:4]]
DATA_CANDIDATES = []
for base in SEARCH_ROOTS:
    DATA_CANDIDATES.extend([
        base / "data" / "processed" / "buyer_clustered_dataset.csv",
        base / "data" / "buyer_clustered_dataset.csv",
        base / "data" / "raw" / "buyer_clustered_dataset.csv",
        base / "data" / "processed" / "buyer_analytical_dataset.csv",
        base / "data" / "buyer_analytical_dataset.csv",
        base / "data" / "raw" / "buyer_analytical_dataset.csv",
    ])
REPORTS = next((base / "outputs" / "reports" for base in SEARCH_ROOTS if (base / "outputs" / "reports").is_dir()), ROOT / "outputs" / "reports")
FIGURES = next((base / "outputs" / "figures" for base in SEARCH_ROOTS if (base / "outputs" / "figures").is_dir()), ROOT / "outputs" / "figures")

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.main-title {font-size: clamp(1.7rem, 3vw, 2.5rem); font-weight: 800; line-height: 1.15;}
.subtitle {font-size: 1rem; opacity: .78; margin: .35rem 0 1.2rem;}
.owner-card {padding: 13px 15px; border-radius: 12px; border: 1px solid rgba(128,128,128,.3); margin: 10px 0;}
.section-title {font-size: 1.45rem; font-weight: 750; margin: .5rem 0;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# Live clock stays independent from the dashboard/data loading.
components.html("""
<div style="font-family:Arial,sans-serif;text-align:center;padding:10px;border:1px solid #8885;border-radius:12px">
 <div style="font-size:12px;font-weight:700;color:#00bcd4">LIVE CLOCK • INDIA STANDARD TIME</div>
 <div id="clock" style="font-size:27px;font-weight:800;color:#00bcd4">--:--:--</div>
 <div id="date" style="font-size:12px">Loading date…</div>
</div>
<script>
function tick(){
 const now=new Date();
 document.getElementById('clock').textContent=now.toLocaleTimeString('en-IN',{timeZone:'Asia/Kolkata',hour12:false});
 document.getElementById('date').textContent=now.toLocaleDateString('en-IN',{timeZone:'Asia/Kolkata',weekday:'long',day:'2-digit',month:'long',year:'numeric'});
}
tick(); setInterval(tick,1000);
</script>
""", height=105)

st.sidebar.markdown("## 🏠 REAL ESTATE AI")
st.sidebar.markdown("<div class='owner-card'><b>OWNER</b><br>Rohit Kumar Mohanty</div>", unsafe_allow_html=True)
st.sidebar.caption("Buyer Segmentation & Investment Profiling")

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

data_path = next((candidate for candidate in DATA_CANDIDATES if candidate.is_file()), None)

# Fallback search includes sibling/parent repository folders. This specifically
# supports a repo layout where app.py is in code/ but data/ is at repository root.
if data_path is None:
    wanted_names = {"buyer_clustered_dataset.csv", "buyer_analytical_dataset.csv"}
    matches = []
    for base in SEARCH_ROOTS:
        try:
            matches.extend(
                p for p in base.rglob("*.csv")
                if p.is_file()
                and p.name.lower() in wanted_names
                and not any(part.lower() in {".git", ".venv", "venv", "site-packages"} for part in p.parts)
            )
        except (OSError, PermissionError):
            continue
    if matches:
        matches = list(set(matches))
        matches.sort(key=lambda p: (p.name.lower() != "buyer_clustered_dataset.csv", len(p.parts)))
        data_path = matches[0]

if data_path is None:
    st.error("The processed buyer dataset could not be found in the deployed project files.")
    st.markdown(
        "The app checked `data/`, `data/processed/`, and nested project folders for "
        "`buyer_clustered_dataset.csv` or `buyer_analytical_dataset.csv`."
    )
    # Show deployment diagnostics instead of guessing, so any remaining path issue
    # is immediately visible in the hosted app.
    st.caption(f"App directory detected by Streamlit: `{ROOT}`")
    st.caption("Repository folders checked: " + ", ".join(f"`{base}`" for base in SEARCH_ROOTS))
    visible = []
    for base in SEARCH_ROOTS:
        try:
            visible.extend(str(p) for p in base.rglob("*.csv") if p.is_file() and not any(part.lower() in {".git", ".venv", "venv", "site-packages"} for part in p.parts))
        except (OSError, PermissionError):
            continue
    if visible:
        st.write("CSV files visible to the deployed app:")
        st.code("\n".join(visible[:60]))
    else:
        st.warning("No CSV files are visible in the app folder or its parent repository folders.")
    st.stop()

try:
    source_df = load_dataset(str(data_path))
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

if source_df.empty:
    st.error("The dataset exists but contains no rows.")
    st.stop()

df = source_df.copy()
st.markdown("<div class='main-title'>🏠 Real Estate Buyer Segmentation & Investment Profiling</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Machine-learning-powered market intelligence, buyer behavior, and investment analytics</div>", unsafe_allow_html=True)
st.caption("Dashboard Owner: Rohit Kumar Mohanty  ·  Models: K-Means and Hierarchical Clustering")

# Filters are applied consistently across every dashboard section.
st.sidebar.markdown("---")
st.sidebar.header("🔎 Dashboard Filters")
filter_cols = ["country", "region", "acquisition_purpose", "client_type", "gender", "loan_applied", "referral_channel"]
for col in filter_cols:
    if col in df.columns:
        values = sorted(df[col].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(col.replace("_", " ").title(), values, key=f"filter_{col}")
        if selected:
            df = df[df[col].astype(str).isin(selected)]
if "kmeans_cluster" in df.columns:
    clusters = sorted(df["kmeans_cluster"].dropna().unique().tolist())
    selected_clusters = st.sidebar.multiselect("K-Means Cluster", clusters, format_func=lambda x: f"Cluster {x}", key="filter_cluster")
    if selected_clusters:
        df = df[df["kmeans_cluster"].isin(selected_clusters)]

if df.empty:
    st.warning("No buyers match the selected filters. Clear one or more filters in the sidebar.")
    st.stop()

def money(value):
    return f"${value:,.0f}" if pd.notna(value) else "N/A"

total_buyers = len(df)
properties = df["properties_purchased"].sum() if "properties_purchased" in df else 0
investment = df["total_investment"].sum() if "total_investment" in df else 0
satisfaction = df["satisfaction_score"].mean() if "satisfaction_score" in df else float("nan")
avg_investment = df["total_investment"].mean() if "total_investment" in df else 0
purchased_buyers = int((df["properties_purchased"] > 0).sum()) if "properties_purchased" in df else 0

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("👥 Buyers", f"{total_buyers:,}")
m2.metric("🏘️ Properties Purchased", f"{int(properties):,}")
m3.metric("💰 Total Investment", money(investment))
m4.metric("📊 Avg. Investment / Buyer", money(avg_investment))
m5.metric("⭐ Avg. Satisfaction", f"{satisfaction:.2f}/5" if pd.notna(satisfaction) else "N/A")

overview, behavior_tab, geography_tab, segments_tab, data_tab, reports_tab = st.tabs([
    "📈 Overview", "🧠 Buyer Behavior", "🌍 Geography", "🎯 Segments", "📋 Data Explorer", "📁 Model Reports"
])

with overview:
    st.markdown("<div class='section-title'>Market Overview</div>", unsafe_allow_html=True)
    a,b = st.columns(2)
    if "kmeans_cluster" in df:
        counts = df["kmeans_cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Buyers")
        a.plotly_chart(px.bar(counts, x="Cluster", y="Buyers", color="Cluster", text="Buyers",
                              title="Buyer Distribution by K-Means Cluster"), use_container_width=True)
    if "acquisition_purpose" in df:
        purpose = df["acquisition_purpose"].value_counts().rename_axis("Purpose").reset_index(name="Buyers")
        b.plotly_chart(px.pie(purpose, names="Purpose", values="Buyers", hole=.42,
                              title="Acquisition Purpose Mix"), use_container_width=True)
    c,d = st.columns(2)
    if "client_type" in df:
        client_mix = df["client_type"].value_counts().rename_axis("Client Type").reset_index(name="Buyers")
        c.plotly_chart(px.bar(client_mix, x="Client Type", y="Buyers", color="Client Type",
                              title="Individual vs Company Buyers"), use_container_width=True)
    if "satisfaction_score" in df and "acquisition_purpose" in df:
        sat = df.groupby("acquisition_purpose", as_index=False)["satisfaction_score"].mean()
        d.plotly_chart(px.bar(sat, x="acquisition_purpose", y="satisfaction_score",
                              title="Average Satisfaction by Acquisition Purpose",
                              labels={"acquisition_purpose":"Purpose","satisfaction_score":"Avg. Satisfaction"}),
                       use_container_width=True)
    st.markdown("#### Key takeaways")
    st.write(f"- **{purchased_buyers:,}** buyers in the current selection have at least one recorded purchase.")
    if "acquisition_purpose" in df and not df.empty:
        top_purpose = df["acquisition_purpose"].mode()
        if not top_purpose.empty:
            st.write(f"- Most common acquisition purpose: **{top_purpose.iloc[0]}**.")
    if "region" in df and not df.empty:
        top_region = df["region"].mode()
        if not top_region.empty:
            st.write(f"- Most represented region: **{top_region.iloc[0]}**.")

with behavior_tab:
    st.markdown("<div class='section-title'>Investor Behavior & Spending</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    if "total_investment" in df:
        c1.plotly_chart(px.histogram(df, x="total_investment", nbins=30, title="Distribution of Total Buyer Investment",
                                    labels={"total_investment":"Total Investment"}), use_container_width=True)
    if "average_property_price" in df:
        c2.plotly_chart(px.box(df, x="acquisition_purpose", y="average_property_price", color="acquisition_purpose",
                               title="Average Property Price by Purpose"), use_container_width=True)
    c3,c4 = st.columns(2)
    if {"properties_purchased","total_investment"}.issubset(df.columns):
        c3.plotly_chart(px.scatter(df, x="properties_purchased", y="total_investment",
                                   color="kmeans_cluster" if "kmeans_cluster" in df else None,
                                   hover_data=[c for c in ["client_id","region","acquisition_purpose"] if c in df.columns],
                                   title="Purchase Count vs Total Investment"), use_container_width=True)
    if {"kmeans_cluster","total_investment","properties_purchased","average_property_price","satisfaction_score"}.issubset(df.columns):
        behavior = df.groupby("kmeans_cluster").agg(
            Buyers=("client_id","count"), Avg_Investment=("total_investment","mean"),
            Avg_Properties=("properties_purchased","mean"), Avg_Property_Price=("average_property_price","mean"),
            Avg_Satisfaction=("satisfaction_score","mean")
        ).reset_index()
        c4.dataframe(behavior.round(2), use_container_width=True, hide_index=True)
    if "loan_applied" in df:
        loan = df["loan_applied"].value_counts().rename_axis("Loan Applied").reset_index(name="Buyers")
        st.plotly_chart(px.pie(loan, names="Loan Applied", values="Buyers", hole=.4,
                               title="Loan Application Profile"), use_container_width=True)

with geography_tab:
    st.markdown("<div class='section-title'>Geographic Buyer Analysis</div>", unsafe_allow_html=True)
    g1,g2 = st.columns(2)
    if "region" in df:
        region_counts = df["region"].value_counts().head(20).rename_axis("Region").reset_index(name="Buyers")
        g1.plotly_chart(px.bar(region_counts.sort_values("Buyers"), x="Buyers", y="Region", orientation="h",
                               title="Top Regions by Buyer Count"), use_container_width=True)
    if "country" in df:
        country_counts = df["country"].value_counts().rename_axis("Country").reset_index(name="Buyers")
        g2.plotly_chart(px.pie(country_counts, names="Country", values="Buyers", hole=.4,
                               title="Buyers by Country"), use_container_width=True)
    if {"region","total_investment"}.issubset(df.columns):
        region_invest = df.groupby("region", as_index=False)["total_investment"].sum().sort_values("total_investment", ascending=False).head(20)
        st.plotly_chart(px.bar(region_invest, x="region", y="total_investment",
                               title="Total Investment by Region", labels={"region":"Region","total_investment":"Total Investment"}),
                        use_container_width=True)

with segments_tab:
    st.markdown("<div class='section-title'>Segment Insights</div>", unsafe_allow_html=True)
    if "kmeans_cluster" in df:
        agg = {}
        if "client_id" in df: agg["client_id"] = "count"
        for col in ["age","satisfaction_score","properties_purchased","total_investment","average_property_price"]:
            if col in df: agg[col] = "mean"
        summary = df.groupby("kmeans_cluster").agg(agg).reset_index().rename(columns={"client_id":"buyers"})
        st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
        s1,s2 = st.columns(2)
        if {"kmeans_cluster","total_investment"}.issubset(df.columns):
            s1.plotly_chart(px.box(df, x="kmeans_cluster", y="total_investment", color="kmeans_cluster",
                                   title="Investment by K-Means Cluster"), use_container_width=True)
        if {"kmeans_cluster","acquisition_purpose"}.issubset(df.columns):
            cross = pd.crosstab(df["kmeans_cluster"], df["acquisition_purpose"]).reset_index().melt(
                id_vars="kmeans_cluster", var_name="Purpose", value_name="Buyers")
            s2.plotly_chart(px.bar(cross, x="kmeans_cluster", y="Buyers", color="Purpose", barmode="stack",
                                   title="Acquisition Purpose within Each Cluster"), use_container_width=True)
    if "hierarchical_cluster" in df:
        hcounts = df["hierarchical_cluster"].value_counts().sort_index().rename_axis("Hierarchical Cluster").reset_index(name="Buyers")
        st.plotly_chart(px.bar(hcounts, x="Hierarchical Cluster", y="Buyers", color="Hierarchical Cluster",
                               title="Hierarchical Clustering Distribution"), use_container_width=True)

with data_tab:
    st.markdown("<div class='section-title'>Buyer Data Explorer</div>", unsafe_allow_html=True)
    st.write(f"Showing **{len(df):,}** rows after filters.")
    search = st.text_input("Search by client ID, name, country, or region", placeholder="e.g. C0001 or California")
    view = df.copy()
    if search.strip():
        mask = pd.Series(False, index=view.index)
        for col in ["client_id","first_name","last_name","country","region"]:
            if col in view:
                mask |= view[col].astype(str).str.contains(search.strip(), case=False, na=False)
        view = view[mask]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered data as CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name="real_estate_buyer_data.csv", mime="text/csv")

with reports_tab:
    st.markdown("<div class='section-title'>Saved Pipeline Reports & EDA Figures</div>", unsafe_allow_html=True)
    report_files = [("Cluster Summary", REPORTS / "cluster_summary.csv"),
                    ("Cluster Evaluation", REPORTS / "cluster_evaluation.csv"),
                    ("Pipeline Summary", REPORTS / "pipeline_summary.txt")]
    found_report = False
    for label, path in report_files:
        if path.exists():
            found_report = True
            st.markdown(f"#### {label}")
            if path.suffix == ".csv":
                report_df = pd.read_csv(path)
                st.dataframe(report_df, use_container_width=True, hide_index=True)
                st.download_button(f"Download {label}", report_df.to_csv(index=False).encode("utf-8"),
                                   file_name=path.name, mime="text/csv", key=f"download_{path.name}")
            else:
                st.code(path.read_text(encoding="utf-8"))
    if not found_report:
        st.info("No saved reports found. The main dashboard still works from the processed dataset.")
    image_names = ["elbow_method.png","silhouette_scores.png","client_type.png","acquisition_purpose.png","total_investment_distribution.png","silhouette_scores.png"]
    existing_images = [FIGURES / name for name in dict.fromkeys(image_names) if (FIGURES / name).exists()]
    if existing_images:
        cols = st.columns(2)
        for i, path in enumerate(existing_images):
            with cols[i % 2]:
                st.image(str(path), caption=path.stem.replace("_"," ").title(), use_container_width=True)

st.markdown("---")
st.caption("Real Estate Buyer Intelligence · K-Means + Hierarchical Clustering · Owner: Rohit Kumar Mohanty")
