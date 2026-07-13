import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from streamlit_option_menu import option_menu

#1. PAGE CONFIGURATION & STYLING

st.set_page_config(
    page_title="DHDS Model Analytics & Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Dynamic App Background */
    .stApp {
        background-color: #0f172a;
        background-image: 
            radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
        background-attachment: fixed;
    }
    
    /* Premium Glassmorphic Cards */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-left: 4px solid #3b82f6 !important;
        padding: 24px !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s ease !important;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 20px 40px -10px rgba(59, 130, 246, 0.3) !important;
        border-left-color: #8b5cf6 !important;
    }
    
    /* Ultimate Form Styling */
    div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%) !important;
        backdrop-filter: blur(20px) !important;
        padding: 2.5rem !important;
        border-radius: 20px !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Input Styling */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    div[data-baseweb="select"] > div:hover, 
    div[data-baseweb="input"] > div:hover,
    div[data-baseweb="select"] > div:focus-within, 
    div[data-baseweb="input"] > div:focus-within {
        border-color: rgba(59, 130, 246, 0.6) !important;
        background-color: rgba(30, 41, 59, 0.8) !important;
        box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.6) !important;
    }
    
    /* Typography */
    label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px !important;
    }
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* Glowing Gradient Submit Button */
    button[kind="primary"] {
        width: 100% !important;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 10px 25px -5px rgba(139, 92, 246, 0.5) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-top: 1.5rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    button[kind="primary"]:hover {
        transform: translateY(-3px) scale(1.01) !important;
        box-shadow: 0 20px 35px -5px rgba(139, 92, 246, 0.7) !important;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%) !important;
    }
    
    /* Export Button Styling */
    div[data-testid="stDownloadButton"] > button {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: rgba(59, 130, 246, 0.2) !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    /* Fit to screen adjustments */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 95% !important;
    }
    </style>
    """, unsafe_allow_html=True)


#2. DATA LOADING & NOTEBOOK LOGIC REPLICATION

@st.cache_data
def load_and_process_data():
    """
    Loads and preprocesses the CDC DHDS dataset.
    Optimized to load only necessary columns, significantly reducing the RAM footprint 
    and improving dashboard performance.
    """
    try:
        # Define only the columns we actually need for the UI and the Model
        cols_to_use = ['Year', 'LocationAbbr', 'LocationDesc', 'Indicator', 
                       'Response', 'Stratification1', 'Stratification2', 
                       'Data_Value', 'WeightedNumber', 'Latitude', 'Longitude']
        
        # Read CSV with a column filter to save memory
        df = pd.read_csv("DHDS_cleaned.csv", usecols=lambda c: c in cols_to_use)
    except FileNotFoundError:
        st.error("File 'DHDS_cleaned.csv' not found. Please ensure it is in the directory.")
        return None, None

    # Filter out rows that are missing critical features required by the Random Forest model
    feature_cols = ['Year', 'LocationAbbr', 'Indicator', 'Stratification1', 'Response', 'Stratification2']
    df_model = df.dropna(subset=feature_cols + ['Data_Value']).copy()
    
    return df, df_model

df_raw, df_model = load_and_process_data()

# --- NEW CACHED MODEL FOR PREDICTION ENGINE ---
@st.cache_resource(show_spinner="Training advanced Random Forest model...")
def train_advanced_model(df):
    feature_cols = ['Year', 'LocationAbbr', 'Indicator', 'Stratification1', 'Response', 'Stratification2']
    X = df[feature_cols].copy()
    y = df['Data_Value']
    
    # We will use WeightedNumber for sample weights, defaulting to 1 if missing/na
    weights = df['WeightedNumber'].fillna(1).values
    
    # Preprocessing: Year is numeric, others are categorical
    categorical_cols = ['LocationAbbr', 'Indicator', 'Stratification1', 'Response', 'Stratification2']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Year']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    
    rf_pipeline.fit(X, y, model__sample_weight=weights)
    return rf_pipeline
    
advanced_model = None
if df_model is not None:
    advanced_model = train_advanced_model(df_model)


#3. SIDEBAR NAVIGATION

with st.sidebar:
    st.markdown("""
        <div style='padding: 0.5rem 0 0.5rem 5px; margin-bottom: 1.5rem;'>
            <h1 style='color: #3b82f6; font-weight: 700; font-size: 2rem; margin: 0; letter-spacing: -1px; display: flex; align-items: center;'>
                <span style='margin-right: 0.5rem; font-size: 1.8rem;'>🧬</span> PredHA
            </h1>
            <p style='color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem; margin-bottom: 0; font-weight: 300;'>Health Analytics Engine</p>
        </div>
    """, unsafe_allow_html=True)
    
    page = option_menu(
        menu_title=None,
        options=["About the Project", "Dashboard Overview", "Prediction Engine", "Geographical Analysis"],
        icons=["info-circle", "bar-chart-line-fill", "robot", "globe-americas"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#3b82f6", "font-size": "1.1rem", "margin-right": "10px"}, 
            "nav-link": {"font-size": "0.95rem", "text-align": "left", "margin":"5px 0", "color": "#cbd5e1", "border-radius": "8px", "padding": "10px 15px"},
            "nav-link-selected": {"background-color": "rgba(59, 130, 246, 0.15)", "color": "#ffffff", "border-left": "4px solid #3b82f6", "font-weight": "600"},
        }
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='font-size: 0.75rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; padding-left: 5px;'>Export Data</p>", unsafe_allow_html=True)
    st.download_button("Download Raw Dataset", df_raw.to_csv(index=False).encode('utf-8'), "DHDS_cleaned.csv", "text/csv", use_container_width=True)


#4. PAGE: ABOUT THE PROJECT

if page == "About the Project":
    st.title("📘 About the Project")
    st.markdown("---")
    
    st.markdown("### 🎯 Project Goal")
    st.info("The primary goal of this application is to analyze and predict the **Prevalence (%)** of various health indicators among adults with disabilities across the United States. By leveraging the CDC's Disability and Health Data System (DHDS), this tool provides policymakers, researchers, and public health officials with actionable, AI-driven insights to better allocate resources and identify demographic health disparities.")
    
    st.markdown("### 📊 Data Dictionary")
    st.markdown("""
    The application utilizes historical data categorized into the following critical fields:
    - **Year:** The year the survey data was collected (e.g., 2016).
    - **Location (LocationAbbr):** The US State or Territory where the respondent resides.
    - **Indicator:** The specific health condition, behavior, or outcome being measured (e.g., *Obesity*, *Depression*, *Aerobic Physical Activity*).
    - **Response:** The categorized answer to the indicator survey (e.g., *Yes*, *No*, *Inactive*, *Sufficiently Active*).
    - **Stratification 1 & 2:** Demographic cuts applied to the data. This allows for deep cross-sectional analysis (e.g., filtering by *Age Group: 18-24* or *Disability Type: Mobility*).
    - **Data_Value (Target):** The calculated prevalence percentage for the specific demographic intersection.
    """)
    
    st.markdown("### 🤖 Technical Methodology")
    st.success("""
    **Machine Learning Architecture:**
    To accurately forecast prevalence, this application implements a **Random Forest Regressor** pipeline. 
    - **Why Random Forest?** Health data demographics often possess complex, non-linear interactions (e.g., the compounded effect of being elderly *and* having a specific disability on obesity rates). Random Forest naturally handles these non-linearities and categorical variables without requiring extreme polynomial feature engineering.
    - **Optimization:** The model is trained dynamically using CDC population weights (`WeightedNumber`) to ensure predictions prioritize demographically significant data blocks, outputting highly accurate, real-world estimates.
    """)


#5. PAGE: DASHBOARD OVERVIEW

elif page == "Dashboard Overview":
    st.title("📊 Executive Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df_raw):,}")
    col2.metric("Locations", df_raw['LocationDesc'].nunique())
    col3.metric("Indicators", df_raw['Indicator'].nunique())
    col4.metric("Median Prevalence", f"{df_raw['Data_Value'].median():.1f}%")
    
    st.markdown("---")
    
    # Interactive Filtering for Dashboard
    st.subheader("Deep Dive Analysis")
    c1, c2, c3 = st.columns(3)
    dash_ind = c1.selectbox("Select Indicator", sorted(df_raw['Indicator'].dropna().unique()), index=0)
    
    # Filter data based on indicator
    dash_df = df_raw[df_raw['Indicator'] == dash_ind]
    
    dash_resp = c2.selectbox("Select Response", sorted(dash_df['Response'].dropna().unique()), index=0)
    dash_strat = c3.selectbox("Select Stratification", sorted(dash_df['Stratification1'].dropna().unique()), index=0)
    
    filtered_dash = dash_df[(dash_df['Response'] == dash_resp) & (dash_df['Stratification1'] == dash_strat)]
    
    if not filtered_dash.empty:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Time Series Trend
            trend_df = filtered_dash.groupby('Year')['Data_Value'].mean().reset_index()
            fig_trend = px.line(trend_df, x='Year', y='Data_Value', markers=True, 
                                title=f"National Average Trend over Time",
                                labels={'Data_Value': 'Average Prevalence (%)'},
                                template="plotly_dark", height=380)
            fig_trend.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with col_chart2:
            # Top 5 / Bottom 5 States for the latest year
            latest_year = filtered_dash['Year'].max()
            year_df = filtered_dash[filtered_dash['Year'] == latest_year]
            state_avg = year_df.groupby('LocationAbbr')['Data_Value'].mean().reset_index()
            state_avg = state_avg.sort_values(by='Data_Value', ascending=False)
            
            top_5 = state_avg.head(5).copy()
            top_5['Type'] = 'Top 5 (Highest)'
            bot_5 = state_avg.tail(5).copy()
            bot_5['Type'] = 'Bottom 5 (Lowest)'
            
            extremes_df = pd.concat([top_5, bot_5])
            fig_extremes = px.bar(extremes_df, x='LocationAbbr', y='Data_Value', color='Type',
                                  title=f"Extremes by State in {latest_year}",
                                  labels={'LocationAbbr': 'State', 'Data_Value': 'Prevalence (%)'},
                                  template="plotly_dark", height=380)
            fig_extremes.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_extremes, use_container_width=True)
    else:
        st.info("No data available for the selected combination.")


#5. PAGE: PREDICTION ENGINE (REGRESSION)

elif page == "Prediction Engine":
    st.title("🤖 Prediction Engine")
    st.markdown("Predict the **prevalence (%)** of various health indicators based on demographics, location, and year using an advanced Random Forest model.")
    
    if advanced_model is None:
        st.error("Model could not be loaded. Please ensure data is present.")
    else:
        with st.container(border=True):
            st.markdown("### ⚙️ Configuration Panel")
            st.markdown("<p style='color: #94a3b8; font-size: 0.9rem; margin-top: -10px; margin-bottom: 20px;'>Select the parameters below to generate an AI-powered prevalence prediction.</p>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3, gap="large")
            with c3:
                input_ind = st.selectbox("Indicator", sorted(df_raw['Indicator'].dropna().unique()), help="The health condition or risk behavior.")
                
                valid_responses = sorted(df_raw[df_raw['Indicator'] == input_ind]['Response'].dropna().unique())
                input_resp = st.selectbox("Response", valid_responses, help="The specific response category.")
                
            with c1:
                input_year = st.number_input("Year", value=2016, help="The year for prediction.")
                
                valid_strat1 = sorted(df_raw[(df_raw['Indicator'] == input_ind) & (df_raw['Response'] == input_resp)]['Stratification1'].dropna().unique())
                if not valid_strat1: valid_strat1 = ["None"]
                input_strat1 = st.selectbox("Stratification 1", valid_strat1, help="Primary demographic filter.")
                
            with c2:
                state_map = df_raw[['LocationDesc', 'LocationAbbr']].drop_duplicates().set_index('LocationDesc')['LocationAbbr'].to_dict()
                loc_name = st.selectbox("Location", sorted(state_map.keys()), help="State or Territory.")
                
                valid_strat2 = sorted(df_raw[(df_raw['Indicator'] == input_ind) & (df_raw['Response'] == input_resp) & (df_raw['Stratification1'] == input_strat1)]['Stratification2'].dropna().unique())
                if not valid_strat2: valid_strat2 = ["None"]
                input_strat2 = st.selectbox("Stratification 2", valid_strat2, help="Secondary demographic filter.")
                
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.button("Generate Prediction", type="primary", use_container_width=True)
            
            if submit_button:
                try:
                    # Create input dataframe matching training structure
                    input_df = pd.DataFrame([{
                        'Year': input_year,
                        'LocationAbbr': state_map[loc_name],
                        'Indicator': input_ind,
                        'Stratification1': input_strat1,
                        'Response': input_resp,
                        'Stratification2': input_strat2
                    }])
                    
                    pred = advanced_model.predict(input_df)[0]
                    # Clip prediction between 0 and 100
                    pred_clipped = max(0.0, min(100.0, pred))
                    
                    st.markdown("---")
                    st.markdown("<h4 style='color: #4ade80;'>✅ Prediction Generated Successfully!</h4>", unsafe_allow_html=True)
                    
                    col_metric, col_info = st.columns([1, 2], gap="large")
                    with col_metric:
                        st.metric("Predicted Prevalence", f"{pred_clipped:.2f} %")
                    with col_info:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.info("💡 **Model Insight**\n\nThis prediction uses a Random Forest Regressor weighted by CDC population data, accounting for complex non-linear demographic interactions.")
                except Exception as e:
                    st.error("An error occurred while generating the prediction. Please verify your input parameters.")


#6. PAGE: GEOGRAPHICAL ANALYSIS

elif page == "Geographical Analysis":
    st.title("🌍 Geographical Analysis")
    st.markdown("Visualize health data prevalence across the United States.")
    
    if 'Latitude' in df_raw.columns:
        # Filters to make the map meaningful
        c1, c2, c3 = st.columns(3)
        map_ind = c1.selectbox("Indicator for Map", sorted(df_raw['Indicator'].dropna().unique()), index=0)
        
        map_df_ind = df_raw[df_raw['Indicator'] == map_ind]
        map_resp = c2.selectbox("Response for Map", sorted(map_df_ind['Response'].dropna().unique()), index=0)
        map_strat = c3.selectbox("Stratification for Map", sorted(map_df_ind['Stratification1'].dropna().unique()), index=0)
        
        # Filter for map
        final_map_df = map_df_ind[(map_df_ind['Response'] == map_resp) & (map_df_ind['Stratification1'] == map_strat)]
        final_map_df = final_map_df.dropna(subset=['Latitude', 'Longitude'])
        
        if not final_map_df.empty:
            # We aggregate by State and Year just in case there are multiple entries per state for the exact same strat
            agg_map_df = final_map_df.groupby(['Year', 'LocationAbbr', 'LocationDesc', 'Latitude', 'Longitude'])['Data_Value'].mean().reset_index()
            agg_map_df = agg_map_df.sort_values(by='Year')
            
            fig_map = px.choropleth(
                agg_map_df, 
                locations='LocationAbbr',
                locationmode="USA-states",
                color="Data_Value",
                hover_name="LocationDesc",
                animation_frame="Year",
                scope="usa",
                color_continuous_scale="Plasma",
                title="Spatial Distribution over Time",
                labels={'Data_Value': 'Prevalence (%)'},
                template="plotly_dark", height=500
            )
            
            # Improve layout
            fig_map.update_layout(
                geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='#1e293b', landcolor='#334155', subunitcolor='#475569'),
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("No geographical data available for this specific combination.")
    else:
        st.warning("Coordinates not found in the dataset.")
