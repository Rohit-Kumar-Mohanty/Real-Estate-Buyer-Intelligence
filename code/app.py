from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "buyer_clustered_dataset.csv"

st.set_page_config(page_title="Rohit Kumar Mohanty | Real Estate Buyer Intelligence", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.main-title{font-size:34px;font-weight:800;margin-bottom:0}.subtitle{font-size:16px;opacity:.75;margin-top:3px;margin-bottom:20px}
.owner-card{padding:14px;border-radius:12px;border:1px solid rgba(128,128,128,.25);margin-bottom:15px}.section-title{font-size:24px;font-weight:700;margin-top:12px}
</style>""", unsafe_allow_html=True)

components.html("""
<style>
    body {
        margin: 0;
        background: transparent !important;
        font-family: Arial, sans-serif;
    }

    .clock-box {
        text-align: center;
        padding: 10px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.30);
    }

    .clock-label {
        font-size: 13px;
        font-weight: 600;
        color: #00E5FF !important;
    }

    #clock {
        font-size: 28px;
        font-weight: 700;
        color: #00E5FF !important;
    }

    #date {
        font-size: 13px;
        font-weight: 500;
        color: #FFFFFF !important;
    }
</style>

<div class="clock-box">
    <div class="clock-label">LIVE CLOCK • IST</div>
    <div id="clock">--:--:--</div>
    <div id="date">Loading...</div>
</div>

<script>
function updateClock() {
    const n = new Date();

    document.getElementById("clock").innerText =
        n.toLocaleTimeString("en-IN", {
            timeZone: "Asia/Kolkata",
            hour12: false,
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit"
        });

    document.getElementById("date").innerText =
        n.toLocaleDateString("en-IN", {
            timeZone: "Asia/Kolkata",
            weekday: "long",
            day: "2-digit",
            month: "long",
            year: "numeric"
        });
}

updateClock();
setInterval(updateClock, 1000);
</script>
""", height=115)

st.sidebar.markdown("## 🏠 REAL ESTATE AI")
st.sidebar.markdown("<div class='owner-card'><b>OWNER</b><br>Rohit Kumar Mohanty</div>", unsafe_allow_html=True)
st.sidebar.caption("Buyer Segmentation & Investment Profiling")

if not DATA.exists():
    st.warning("Processed model data is not available yet. Open the VS Code terminal and run: `python run_pipeline.py`")
    st.stop()

df = pd.read_csv(DATA)
st.markdown("<div class='main-title'>🏠 Real Estate Buyer Segmentation & Investment Profiling</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Machine Learning Based Market Intelligence Dashboard</div>", unsafe_allow_html=True)
st.markdown("**Dashboard Owner:** Rohit Kumar Mohanty  |  **Model:** K-Means + Hierarchical Clustering")

st.sidebar.markdown("---")
st.sidebar.header("🔎 Filters")
for col in ["country","region","acquisition_purpose","client_type"]:
    if col in df.columns:
        vals = sorted(df[col].dropna().astype(str).unique())
        selected = st.sidebar.multiselect(col.replace("_"," ").title(), vals)
        if selected: df = df[df[col].astype(str).isin(selected)]

c1,c2,c3,c4 = st.columns(4)
c1.metric("👥 Buyers", f"{len(df):,}")
c2.metric("🏢 Properties Purchased", f"{int(df['properties_purchased'].sum()):,}")
c3.metric("💰 Total Investment", f"${df['total_investment'].sum():,.0f}")
c4.metric("⭐ Avg Satisfaction", f"{df['satisfaction_score'].mean():.2f}")

st.markdown("<div class='section-title'>Buyer Segmentation Overview</div>", unsafe_allow_html=True)
counts=df["kmeans_cluster"].value_counts().sort_index().reset_index(); counts.columns=["cluster","buyers"]
st.plotly_chart(px.bar(counts,x="cluster",y="buyers",text="buyers",title="Buyer Distribution by K-Means Cluster"),use_container_width=True)

st.markdown("<div class='section-title'>Investor Behavior Dashboard</div>", unsafe_allow_html=True)
behavior=df.groupby("kmeans_cluster").agg(Buyers=("client_id","count"),Avg_Investment=("total_investment","mean"),Avg_Properties=("properties_purchased","mean"),Avg_Property_Price=("average_property_price","mean"),Avg_Satisfaction=("satisfaction_score","mean")).reset_index()
st.dataframe(behavior,use_container_width=True,hide_index=True)

st.markdown("<div class='section-title'>Geographic Buyer Analysis</div>", unsafe_allow_html=True)
geo=df["region"].value_counts().reset_index(); geo.columns=["region","buyers"]
st.plotly_chart(px.bar(geo,x="region",y="buyers",text="buyers",title="Buyers by Region"),use_container_width=True)

st.markdown("<div class='section-title'>Segment Insights Panel</div>", unsafe_allow_html=True)
ins=df.groupby("kmeans_cluster").agg(Buyers=("client_id","count"),Avg_Age=("age","mean"),Investment_Buyers=("acquisition_purpose",lambda s:(s.astype(str).str.lower()=="investment").sum()),Loan_Users=("loan_applied",lambda s:(s.astype(str).str.lower()=="yes").sum()),Avg_Total_Investment=("total_investment","mean"),Avg_Satisfaction=("satisfaction_score","mean")).reset_index()
st.dataframe(ins,use_container_width=True,hide_index=True)
st.markdown("---")
st.caption("Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence • Owner: Rohit Kumar Mohanty")
