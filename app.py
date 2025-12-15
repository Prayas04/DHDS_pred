import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             confusion_matrix, roc_curve, auc, mean_squared_error, 
                             r2_score, mean_absolute_error)


#1. PAGE CONFIGURATION & STYLING

st.set_page_config(
    page_title="DHDS Model Analytics & Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)


#2. DATA LOADING & NOTEBOOK LOGIC REPLICATION

@st.cache_data
def load_and_process_data():
    #Load Data
    try:
        df = pd.read_csv("DHDS_cleaned.csv")
    except FileNotFoundError:
        st.error("File 'DHDS_cleaned.csv' not found. Please ensure it is in the directory.")
        return None, None, None, None, None, None

    #REPLICATING MODEL.IPYNB LOGIC
    median_val = df['Data_Value'].median()
    df['Target_Class'] = (df['Data_Value'] > median_val).astype(int)

    feature_cols = ['Year', 'LocationAbbr', 'Indicator', 'Stratification1', 'Response', 'Stratification2']
    df_model = df.dropna(subset=feature_cols + ['Data_Value']).copy()
    
    X = df_model[feature_cols].copy()
    y_class = df_model['Target_Class']
    y_reg = df_model['Data_Value']

    label_encoders = {}
    for col in ['LocationAbbr', 'Indicator', 'Stratification1', 'Response', 'Stratification2']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 

    split_data = train_test_split(X_scaled, y_class, y_reg, test_size=0.2, random_state=42)
    
    #RETURN X_scaled so it can be used outside
    return df, df_model, X_scaled, label_encoders, scaler, split_data

#Unpack X_scaled here
df_raw, df_model, X_scaled, label_encoders, scaler, split_data = load_and_process_data()

if df_raw is not None:
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = split_data


#3. SIDEBAR NAVIGATION

with st.sidebar:
    st.title("🔍 FUNCTIONALITIES")
    page = st.radio("Go to", [
        "Dashboard Overview", 
        "Prediction Engine (Regression)", 
        "Geographical Analysis",
        "Model Performance Metrics",
        "Clustering Analysis (PCA)",
        "Download Data"
    ])


#4. PAGE: DASHBOARD OVERVIEW

if page == "Dashboard Overview":
    st.title("📊 Executive Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df_raw):,}")
    col2.metric("Locations", df_raw['LocationDesc'].nunique())
    col3.metric("Indicators", df_raw['Indicator'].nunique())
    col4.metric("Median Data Value", f"{df_raw['Data_Value'].median():.2f}")
    
    st.plotly_chart(px.histogram(df_raw, x="Data_Value", nbins=50, title="Distribution of Data Values"), use_container_width=True)


#5. PAGE: PREDICTION ENGINE (REGRESSION)

elif page == "Prediction Engine (Regression)":
    st.title("🤖 Prediction Engine")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train_reg)
    
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            input_year = st.number_input("Year", 2000, 2030, 2016)
            state_map = df_raw[['LocationDesc', 'LocationAbbr']].drop_duplicates().set_index('LocationDesc')['LocationAbbr'].to_dict()
            loc_name = st.selectbox("Location", sorted(state_map.keys()))
            input_ind = st.selectbox("Indicator", label_encoders['Indicator'].classes_)
        with c2:
            input_strat1 = st.selectbox("Stratification 1", label_encoders['Stratification1'].classes_)
            input_resp = st.selectbox("Response", label_encoders['Response'].classes_)
            input_strat2 = st.selectbox("Stratification 2", label_encoders['Stratification2'].classes_)
            
        if st.form_submit_button("Predict"):
            try:
                vec = np.array([[
                    input_year,
                    label_encoders['LocationAbbr'].transform([state_map[loc_name]])[0],
                    label_encoders['Indicator'].transform([input_ind])[0],
                    label_encoders['Stratification1'].transform([input_strat1])[0],
                    label_encoders['Response'].transform([input_resp])[0],
                    label_encoders['Stratification2'].transform([input_strat2])[0]
                ]])
                pred = lin_reg.predict(scaler.transform(vec))[0]
                st.metric("Predicted Value", f"{pred:.4f}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")


#6. PAGE: GEOGRAPHICAL ANALYSIS

elif page == "Geographical Analysis":
    st.title("🌍 Geographical Analysis")
    if 'Latitude' in df_raw.columns:
        yr = st.selectbox("Year", sorted(df_raw['Year'].unique(), reverse=True))
        map_df = df_raw[df_raw['Year'] == yr].dropna(subset=['Latitude', 'Longitude'])
        st.plotly_chart(px.scatter_geo(map_df, lat='Latitude', lon='Longitude', color="Data_Value", scope="usa", title=f"Map ({yr})"), use_container_width=True)
    else:
        st.warning("Coordinates not found.")


#7. PAGE: MODEL PERFORMANCE METRICS

elif page == "Model Performance Metrics":
    st.title("📈 Model Performance")
    
    #Added tabs for Regression and Feature Importance
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Classification Metrics", 
        "Confusion Matrix", 
        "ROC Curves", 
        "Regression Metrics", 
        "Feature Importance"
    ])
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(random_state=42, probability=True)
    }
    
    trained_models = {}
    results = []
    
    #Train and evaluate classifiers
    for name, model in models.items():
        model.fit(X_train, y_train_class)
        trained_models[name] = model
        preds = model.predict(X_test)
        
        #ADDED ALL CLASSIFICATION METRICS
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test_class, preds),
            "Precision": precision_score(y_test_class, preds, zero_division=0),
            "Recall": recall_score(y_test_class, preds, zero_division=0),
            "F1-Score": f1_score(y_test_class, preds, zero_division=0)
        })
        
    with tab1:
        st.markdown("### Classifier Performance")
        metrics_df = pd.DataFrame(results).set_index("Model")
        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
        st.plotly_chart(px.bar(metrics_df.reset_index().melt(id_vars="Model"), x="Model", y="value", color="variable", barmode="group"), use_container_width=True)
        
    with tab2:
        sel_model = st.selectbox("Select Model for Matrix", list(models.keys()))
        cm = confusion_matrix(y_test_class, trained_models[sel_model].predict(X_test))
        st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title=f"Confusion Matrix: {sel_model}"), use_container_width=True)
        
    with tab3:
        fig_roc = go.Figure()
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        for name, model in trained_models.items():
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test_class, probs)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc(fpr, tpr):.2f})"))
        fig_roc.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    #ADDED REGRESSION METRICS
    with tab4:
        st.markdown("### Linear Regression Evaluation")
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train_reg)
        reg_preds = lin_reg.predict(X_test)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("MSE", f"{mean_squared_error(y_test_reg, reg_preds):.2f}")
        c2.metric("MAE", f"{mean_absolute_error(y_test_reg, reg_preds):.2f}")
        c3.metric("R2 Score", f"{r2_score(y_test_reg, reg_preds):.4f}")
        
        #Plot Actual vs Predicted
        fig_reg = px.scatter(x=y_test_reg, y=reg_preds, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted Values")
        fig_reg.add_shape(type='line', line=dict(dash='dash', color='red'), x0=y_test_reg.min(), y0=y_test_reg.min(), x1=y_test_reg.max(), y1=y_test_reg.max())
        st.plotly_chart(fig_reg, use_container_width=True)

   
    with tab5:
        st.markdown("### Feature Importance (Decision Tree)")
        dt_model = trained_models["Decision Tree"]
        importances = dt_model.feature_importances_
        feature_names = ['Year', 'LocationAbbr', 'Indicator', 'Stratification1', 'Response', 'Stratification2']
        
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        st.plotly_chart(px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Feature Importance"), use_container_width=True)


#8. PAGE: CLUSTERING ANALYSIS

elif page == "Clustering Analysis (PCA)":
    st.title("🧩 Clustering & PCA")
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_scaled) 
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters.astype(str)
    
    st.plotly_chart(px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title="K-Means (PCA Projection)"), use_container_width=True)


elif page == "Download Data":
    st.title("💾 Data")
    st.download_button("Download CSV", df_raw.to_csv(index=False).encode('utf-8'), "DHDS_cleaned.csv", "text/csv")