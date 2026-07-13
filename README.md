# 🧬 PredCA Health Analytics Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**PredCA** is an ultra-premium, production-grade Machine Learning web dashboard designed to forecast and analyze the prevalence of health indicators among adults with varying disability statuses. Built on top of the Centers for Disease Control and Prevention (CDC) **DHDS** (Disability and Health Data System) dataset.

---

## 🎯 Project Goals

The core objective of this project is to leverage intersectional demographic data (Age, Race, Sex, and specific Disability Types) to accurately predict health outcomes. This tool empowers researchers, policymakers, and public health officials to:
1. **Analyze** historical geographic trends through interactive choropleth mapping.
2. **Predict** future health risk prevalence across specific demographic combinations using a tuned Random Forest pipeline.
3. **Understand** the complex, non-linear relationships between health barriers (e.g., healthcare access, obesity) and disability status.

---

## 🚀 Key Features

- **🧠 Advanced AI Prediction Engine:** Utilizes an empirically justified `RandomForestRegressor` embedded within a highly optimized Scikit-Learn `Pipeline`. The model effectively captures complex, non-linear demographic interactions, vastly outperforming standard linear models.
- **🔄 Dynamic Cascading UI:** The dashboard features a mathematically robust user interface. Selection inputs (Indicators, Responses, Stratifications) are dynamically chain-linked, actively preventing mathematically impossible queries ("Junk-in, Junk-out") from ever reaching the prediction model.
- **🗺️ High-Fidelity Geographical Analysis:** Integrates `plotly.express` choropleth mapping for stunning, animated spatial distributions of health data across the United States.
- **🛡️ Secure & Obfuscated:** Architected to natively prevent SQL Injection and Arbitrary Code Execution. All inputs are strictly bound to dataset-derived selectboxes. Global configurations actively suppress internal tracebacks, ensuring a sleek, unbreakable front-end experience.
- **⚡ Memory Optimized:** Custom data loading pipelines leverage Pandas' `usecols`, drastically reducing the application's RAM footprint for lightning-fast startup and inference times.

---

## ⚙️ Architecture & Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) 
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) 
- **Data Pipeline:** [Pandas](https://pandas.pydata.org/), Numpy.
- **Data Visualization:** [Plotly Express](https://plotly.com/python/plotly-express/), Matplotlib, Seaborn.

---

## 💻 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/PredCA.git
   cd PredCA
   ```

2. **Activate your Virtual Environment:**
   *(Windows)*
   ```bash
   .\venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 📂 Jupyter Notebooks (Research)
This repository includes a suite of strictly optimized Jupyter Notebooks used for the underlying research and model selection:
- `DC.ipynb`: Raw data cleaning, missing value extraction, and strict academic assertions.
- `Visualizations.ipynb`: Advanced Exploratory Data Analysis (EDA), featuring non-destructive skewness transformations (`signed_log1p`).
- `model.ipynb`: The comprehensive Machine Learning pipeline comparison (Logistic Regression, SVM, KNN, Decision Trees) empirically justifying the deployment of the Random Forest algorithm.

---
*Built with professional optimization, rigorous data science standards, and premium design philosophy.*
