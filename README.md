#🏠 Real Estate Buyer Intelligence

Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

A machine learning-based real estate analytics system that identifies meaningful buyer segments and analyzes investment behavior using K-Means Clustering and Hierarchical Clustering.

The project also includes an interactive Streamlit dashboard for exploring buyer demographics, investment behavior, geographic patterns, financing behavior, and cluster-level insights.

👨‍💻 Project Information

Author: Rohit Kumar Mohanty
Project Mentor: Saiprasad Kagne

Domain: Financial Analytics & Real Estate Market Intelligence

Collaborating Organizations:

Unified Mentor

Parcl Co. Limited

📌 Project Overview

Real estate organizations deal with buyers having different investment purposes, financial behaviors, geographic backgrounds, and purchasing patterns.

Traditional buyer analysis may not provide enough insight into these differences.

This project develops a machine learning-based buyer segmentation system that processes real estate buyer and property transaction data to identify groups of buyers with similar characteristics.

The system combines:

Data cleaning

Data integration

Feature engineering

Categorical encoding

Feature scaling

Exploratory Data Analysis

K-Means clustering

Hierarchical clustering

Elbow method

Silhouette analysis

Cluster interpretation

Interactive visualization

The final results are presented through a Streamlit dashboard designed for real estate market intelligence and decision support.

🎯 Project Objectives

The main objectives of this project are:

Analyze real estate buyer characteristics and behavior.

Identify meaningful buyer segments using unsupervised machine learning.

Analyze buyer investment patterns.

Study property purchasing behavior.

Analyze financing and loan application behavior.

Identify geographic differences among buyers.

Compare K-Means and Hierarchical clustering approaches.

Determine an appropriate number of clusters using evaluation metrics.

Provide interactive visualizations through a Streamlit dashboard.

Generate business-oriented insights from the identified buyer groups.

🧩 Problem Statement

Real estate businesses need a better understanding of their buyers in order to improve customer targeting, investment recommendations, and market analysis.

Differences in:

Buyer type

Geographic location

Acquisition purpose

Financing behavior

Property purchases

Investment amount

Demographics

Satisfaction

can make it difficult to create a single strategy for all buyers.

This project addresses this problem by using machine learning-based clustering to identify groups of buyers with similar characteristics.

📊 Dataset

The project works with two primary datasets:

1. Clients Dataset

The client dataset contains buyer-level information including:

client_id

client_type

gender

country

region

date_of_birth

acquisition_purpose

loan_applied

referral_channel

satisfaction_score

2. Properties Dataset

The properties dataset contains real estate property and transaction information.

Property transaction information is integrated with the client dataset using the appropriate client reference.

This allows the project to derive buyer-level investment and purchasing features.

🧹 Data Preparation

The data processing pipeline performs the following operations:

Load client and property datasets.

Remove duplicate records.

Clean categorical values.

Parse date fields.

Convert property sale prices into numerical values.

Identify sold property transactions.

Aggregate property purchases for each buyer.

Calculate buyer-level investment features.

Merge property information with client information.

Handle missing numerical purchase information.

Generate the final analytical dataset.

🛠️ Feature Engineering

The project derives several analytical features from the available data.

Important numerical features include:

Age

Satisfaction Score

Properties Purchased

Total Investment

Average Property Price

Total Floor Area

Average Floor Area

Purchase Span

Property Purchase Indicator

Categorical features include:

Client Type

Gender

Country

Region

Acquisition Purpose

Loan Applied

Referral Channel

🔄 Feature Encoding and Scaling

Categorical variables are converted into machine learning-compatible numerical representations using One-Hot Encoding.

Numerical features are standardized using StandardScaler.

This preprocessing ensures that features with different numerical ranges do not disproportionately influence the clustering algorithm.

🤖 Machine Learning Methodology

The project uses unsupervised machine learning because the buyer groups are not predefined.

K-Means Clustering

K-Means clustering groups buyers based on similarity in their feature representations.

The algorithm attempts to minimize the distance between observations and their assigned cluster centers.

Hierarchical Clustering

Agglomerative Hierarchical Clustering is also applied to analyze buyer grouping from a hierarchical perspective.

Using both methods provides an additional comparison of buyer segmentation patterns.

📐 Cluster Selection

Several K-Means configurations are evaluated using:

Elbow Method

The Elbow Method evaluates the reduction in within-cluster sum of squares as the number of clusters increases.

Silhouette Score

The Silhouette Score evaluates how well observations fit within their assigned clusters compared with other clusters.

The project evaluates multiple cluster configurations and selects the final K-Means configuration based on the clustering evaluation.

📈 Model Results

The final model selected:

3 Buyer Clusters

The selected solution achieved a Silhouette Score of approximately:

0.1670

The dataset contains:

2,000 buyers

7,305 linked sold property purchases

$2,520,750,961 total investment value

Cluster Distribution

Cluster

Number of Buyers

Cluster 0

1,047

Cluster 1

904

Cluster 2

49

The clusters exhibit differences in investment behavior, property purchasing patterns, demographic characteristics, satisfaction, and financing behavior.

Important Observation

Cluster 2 is a relatively small but high-value group. It shows:

Higher average age

Higher average satisfaction

Higher average number of properties purchased

Higher average total investment

Lower proportion of loan users

Cluster 1 generally shows higher investment and property-price characteristics than Cluster 0.

The identified clusters are data-driven segments and should not automatically be treated as predefined business categories without additional validation.

📊 Exploratory Data Analysis

The project generates visualizations for important buyer characteristics, including:

Acquisition Purpose Distribution

Client Type Distribution

Total Investment Distribution

Elbow Method

Silhouette Scores

Cluster Analysis

Generated visualizations are stored inside:

outputs/figures/

🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard provides several analytical modules.

🏠 Buyer Segmentation Overview

Provides an overview of:

Buyer distribution
Cluster distribution
Buyer statistics
Segmentation results

💰 Investor Behavior Dashboard

Analyzes:

Investment patterns
Property purchases
Average property prices
Loan behavior
Buyer purchasing behavior

🌍 Geographic Buyer Analysis

Analyzes buyer distribution across:

Countries
Regions
Geographic segments

📌 Segment Insights Panel

Provides cluster-level insights based on:

Demographics
Investment behavior
Property purchases
Financing behavior
Satisfaction
🔎 Dashboard Filters

Users can interactively filter the dashboard using:

Country
Region
Acquisition Purpose
Client Type

The dashboard also includes a live Indian Standard Time clock.

🧰 Technologies Used

The project is developed using:

Python
Pandas
NumPy
Scikit-learn
Matplotlib
Plotly
Streamlit
Joblib

📁 Project Structure

Real-Estate-Buyer-Intelligence/
│
├── data/
│   ├── raw/
│   │   ├── clients.csv
│   │   └── properties.csv
│   │
│   └── processed/
│
├── docs/
│   ├── STEP_BY_STEP.md
│   ├── PRD_MAPPING.md
│   └── VIVA_QUESTIONS.md
│
├── models/
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── src/
│   ├── init.py
│   ├── data_pipeline.py
│   ├── model.py
│   └── eda.py
│
├── research_paper/
│
├── LICENSE
├── README.md
├── app.py
├── requirements.txt
└── run_pipeline.py


---

## ## ⚙️ Installation

1. Clone the Repository
git clone https://github.com/Rohit-Kumar-Mohanty/Real-Estate-Buyer-Intelligence.git
2. Navigate to the Project Directory
cd Real-Estate-Buyer-Intelligence
3. Create a Virtual Environment Windows
python -m venv .venv
4. Activate the Virtual Environment
.venv\Scripts\activate
5. Install Required Packages
pip install -r requirements.txt

## ▶️ Running the Project

Step 1: Run the Machine Learning Pipeline
python run_pipeline.py

The pipeline performs:

Data cleaning
Data integration
Feature engineering
Exploratory analysis
Feature preprocessing
Cluster evaluation
K-Means training
Hierarchical clustering
Model saving
Output generation

### 🚀 Launch the Streamlit Dashboard

After running the pipeline, execute:

streamlit run app.py

The Streamlit application will open in the browser.

## 📦 Generated Outputs

The project generates:

Processed analytical dataset
Clustered buyer dataset
Cluster summary
EDA visualizations
Elbow plot
Silhouette score plot
Trained K-Means model
Preprocessing pipeline
Hierarchical clustering model
Pipeline summary

These outputs are stored in the relevant project directories.

## 📄 Research Paper

The research paper documents the complete project methodology and findings.

It includes:

Abstract
Introduction
Problem Statement
Objectives
Dataset Description
Data Preparation
Feature Engineering
Machine Learning Methodology
Clustering Evaluation
Results
Cluster Interpretation
Business Insights
Recommendations
Dashboard Description
Limitations
Future Scope
Conclusion
References

The research paper is available in the:

Research Paper/

directory.

## 💡 Business Insights

The segmentation approach can support real estate organizations in:

Identifying groups of similar buyers
Improving customer targeting
Understanding investment behavior
Identifying high-value buyer groups
Studying geographic investment patterns
Understanding financing behavior
Developing more targeted marketing strategies
Supporting personalized investment recommendations

## ⚠️ Limitations

The project has several limitations:

The clustering quality is moderate, as reflected by the Silhouette Score.
The dataset may not capture all factors influencing real estate investment decisions.
Clustering results depend on the selected features and preprocessing approach.
The identified clusters are descriptive and do not establish causal relationships.
The current model does not automatically map clusters to predefined business personas.
Additional external market and economic variables could improve segmentation.
## 🔮 Future Scope

Future versions of the project can include:

Real-time real estate market data
Property location intelligence
External economic indicators
Property price forecasting
Advanced clustering algorithms
DBSCAN and Gaussian Mixture Models
Automated buyer persona generation
Recommendation systems
Interactive geographic maps
Time-series investment analysis
Deep learning-based segmentation
Explainable AI techniques
Deployment on cloud infrastructure
## 🔐 Responsible Data Usage

The project should be used for educational, analytical, and research purposes.

When publishing or sharing datasets publicly, users should ensure that confidential, sensitive, or personally identifiable information is not exposed.

## 📚 Project Documentation

Additional project documentation is available in:

docs/

including:

Step-by-step project instructions
PRD mapping

## 👨‍💻 Author
Rohit Kumar Mohanty

Machine Learning / Cybersecurity Enthusiast

Project: Real Estate Buyer Intelligence

## 👨‍🏫 Project Mentor
Saiprasad Kagne

## 📜 License

This project is provided under the MIT license included in this repository.

## ⭐ Acknowledgement

I would like to express my sincere gratitude to Saiprasad Kagne, my project mentor, for valuable guidance, support, and constructive feedback throughout the development of this project.

I would also like to thank Unified Mentor and Parcl Co. Limited for providing the opportunity and resources to undertake this project.

## 🚀 Project Summary

This project demonstrates how machine learning and interactive data visualization can be combined to analyze real estate buyer behavior and discover meaningful investment segments.

The combination of data preprocessing, feature engineering, clustering, model evaluation, and Streamlit visualization provides an end-to-end framework for real estate market intelligence.
