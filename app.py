import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import requests

# --- CONFIGURATION ---
# ENTER YOUR API KEY HERE
# Get a free key from https://openweathermap.org/api if you want the live weather feature to work perfectly
OPENWEATHER_API_KEY = "YOUR_API_KEY_HERE" 

st.set_page_config(
    page_title="Crop Yield Predictor & Recommender",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DARK MODE CSS ---
# This injects custom CSS to override Streamlit's default look
# It sets the background to dark gray (#121212) and uses green accents (#4CAF50)
st.markdown("""
    <style>
    /* Main Background - Dark Charcoal */
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }
    
    /* Sidebar Background - Slightly Lighter Dark Gray */
    section[data-testid="stSidebar"] {
        background-color: #1E1E1E;
        border-right: 1px solid #333333;
    }
    
    /* Headers - White for Contrast */
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Labels and General Text - Off-white */
    label, .stMarkdown p, .stMarkdown li {
        color: #E0E0E0 !important;
    }

    /* Cards (Metrics/Inputs) - Dark Gray with Green Accent Border */
    .stMetric, div[data-testid="stExpander"] {
        background-color: #262626;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4CAF50; /* Vibrant Green Border */
        box-shadow: 0 2px 5px rgba(0,0,0,0.5);
    }

    /* Metric Value Color */
    div[data-testid="stMetricValue"] {
        color: #FFFFFF;
    }

    /* Input Fields - Dark Background */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #333333;
        color: #FFFFFF;
        border: 1px solid #555555;
    }
    
    /* Buttons - Brighter Agri Green */
    .stButton button {
        background-color: #4CAF50; /* A brighter green */
        color: white !important;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #388E3C; /* Darker green on hover */
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    
    /* Sidebar Nav Items */
    section[data-testid="stSidebar"] .stRadio label {
        color: #E0E0E0;
    }
    
    /* Slider & Widget Accent Colors */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #4CAF50 !important;
    }
    div[data-baseweb="slider"] div[data-testid="stTickBar"] {
        background-color: #4CAF50 !important;
    }
    
    /* Plotly Backgrounds */
    .js-plotly-plot .plotly .bg, .js-plotly-plot .plotly .main-svg {
        background-color: rgba(0,0,0,0) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA & MODEL LOADING ---

@st.cache_data
def load_data():
    """
    Loads the dataset. Tries multiple filenames to be robust.
    Cleans data by removing rows with missing critical info.
    """
    try:
        # Check for various file names
        if os.path.exists('data/crop_yield_final_dataset.csv'):
            df = pd.read_csv('data/crop_yield_final_dataset.csv')
        elif os.path.exists('data/crop_yield_final_dataset (1).csv'):
            df = pd.read_csv('data/crop_yield_final_dataset (1).csv')
        elif os.path.exists('data/final_yield_with_weather.csv'):
            # Fallback to the intermediate file if final one is missing
            df = pd.read_csv('data/final_yield_with_weather.csv')
        else:
            return None
            
        # Basic cleanup
        subset_cols = ['Production', 'Area', 'Rainfall']
        # Only drop if columns exist
        existing_subset = [c for c in subset_cols if c in df.columns]
        df = df.dropna(subset=existing_subset)
        df = df[df['Area'] > 0]
        
        # Calculate Yield if not present
        if 'Yield' not in df.columns:
            df['Yield'] = df['Production'] / df['Area']
            
        # Remove outliers
        df = df[df['Yield'] < 100] 
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_or_train_models(df):
    """
    Tries to load pre-trained .pkl models.
    If missing, it TRAINS them right now using the loaded dataframe.
    """
    files_exist = os.path.exists('model/final_model.pkl') and os.path.exists('model/recommender_model.pkl')
    
    if files_exist:
        # Load pre-trained brains
        with open('model/final_model.pkl', 'rb') as f: regressor = pickle.load(f)
        with open('model/recommender_model.pkl', 'rb') as f: classifier = pickle.load(f)
        with open('model/le_district.pkl', 'rb') as f: le_district = pickle.load(f)
        with open('model/le_season.pkl', 'rb') as f: le_season = pickle.load(f)
        with open('model/le_crop.pkl', 'rb') as f: le_crop = pickle.load(f)
    else:
        st.info("🌱 Models not found on disk. Planting seeds... Training AI models now (this happens once).")
        
        # Initialize Encoders
        le_district = LabelEncoder()
        le_season = LabelEncoder()
        le_crop = LabelEncoder()

        # Transform categorical text to numbers
        df['District_Encoded'] = le_district.fit_transform(df['District_Name'].astype(str))
        df['Season_Encoded'] = le_season.fit_transform(df['Season'].astype(str))
        df['Crop_Encoded'] = le_crop.fit_transform(df['Crop'].astype(str))

        # Check if soil data exists, if not, fill defaults for training to avoid crash
        if 'N' not in df.columns:
            df['N'] = 220
            df['P'] = 20
            df['K'] = 300
            df['pH'] = 7.0

        # --- Train Regressor (Predict Yield) ---
        reg_features = ['District_Encoded', 'Season_Encoded', 'Crop_Encoded', 'Area', 
                        'Rainfall', 'Temperature', 'N', 'P', 'K', 'pH']
        
        # Ensure all features exist
        available_reg_features = [f for f in reg_features if f in df.columns]
        
        X_reg = df[available_reg_features]
        y_reg = df['Yield'] 
        
        regressor = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        regressor.fit(X_reg, y_reg)

        # --- Train Classifier (Recommend Crop) ---
        clf_features = ['District_Encoded', 'Season_Encoded', 'N', 'P', 'K', 'pH', 'Rainfall', 'Temperature']
        available_clf_features = [f for f in clf_features if f in df.columns]
        
        X_clf = df[available_clf_features]
        y_clf = df['Crop_Encoded'] 

        classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        classifier.fit(X_clf, y_clf)

        # Save them for next time (optional, but good practice)
        try:
            with open('model/final_model.pkl', 'wb') as f: pickle.dump(regressor, f)
            with open('model/recommender_model.pkl', 'wb') as f: pickle.dump(classifier, f)
            with open('model/le_district.pkl', 'wb') as f: pickle.dump(le_district, f)
            with open('model/le_season.pkl', 'wb') as f: pickle.dump(le_season, f)
            with open('model/le_crop.pkl', 'wb') as f: pickle.dump(le_crop, f)
        except:
            pass # Ignore save errors on read-only systems
        
    return regressor, classifier, le_district, le_season, le_crop

# Coordinates for Live Weather API mapping
DISTRICT_COORDS = {
    'ARIYALUR': {'lat': 11.1401, 'lon': 79.0786}, 'COIMBATORE': {'lat': 11.0168, 'lon': 76.9558},
    'CUDDALORE': {'lat': 11.7480, 'lon': 79.7714}, 'DHARMAPURI': {'lat': 12.1211, 'lon': 78.1582},
    'DINDIGUL': {'lat': 10.3673, 'lon': 77.9803}, 'ERODE': {'lat': 11.3410, 'lon': 77.7172},
    'KANCHEEPURAM': {'lat': 12.8185, 'lon': 79.6947}, 'KANYAKUMARI': {'lat': 8.0883, 'lon': 77.5385},
    'KARUR': {'lat': 10.9601, 'lon': 78.0766}, 'KRISHNAGIRI': {'lat': 12.5186, 'lon': 78.2137},
    'MADURAI': {'lat': 9.9252, 'lon': 78.1198}, 'NAGAPATTINAM': {'lat': 10.7672, 'lon': 79.8449},
    'NAMAKKAL': {'lat': 11.2148, 'lon': 78.1702}, 'PERAMBALUR': {'lat': 11.2358, 'lon': 78.8810},
    'PUDUKKOTTAI': {'lat': 10.3797, 'lon': 78.8208}, 'RAMANATHAPURAM': {'lat': 9.3639, 'lon': 78.8395},
    'SALEM': {'lat': 11.6643, 'lon': 78.1460}, 'SIVAGANGA': {'lat': 9.8433, 'lon': 78.4809},
    'THANJAVUR': {'lat': 10.7870, 'lon': 79.1378}, 'THE NILGIRIS': {'lat': 11.4102, 'lon': 76.6950},
    'THENI': {'lat': 10.0104, 'lon': 77.4768}, 'THIRUVALLUR': {'lat': 13.1430, 'lon': 79.8954},
    'THIRUVARUR': {'lat': 10.7766, 'lon': 79.6344}, 'THOOTHUKKUDI': {'lat': 8.7642, 'lon': 78.1348},
    'TIRUCHIRAPPALLI': {'lat': 10.7905, 'lon': 78.7047}, 'TIRUNELVELI': {'lat': 8.7139, 'lon': 77.7567},
    'TIRUPPUR': {'lat': 11.1085, 'lon': 77.3411}, 'TIRUVANNAMALAI': {'lat': 12.2253, 'lon': 79.0747},
    'VELLORE': {'lat': 12.9165, 'lon': 79.1325}, 'VILLUPURAM': {'lat': 11.9401, 'lon': 79.4861},
    'VIRUDHUNAGAR': {'lat': 9.5680, 'lon': 77.9624}
}

# Defaults for Soil Doctor
SOIL_DEFAULTS = {
    'THANJAVUR': {'N': 210, 'P': 24, 'K': 310, 'pH': 7.2}, 'COIMBATORE': {'N': 240, 'P': 18, 'K': 550, 'pH': 7.5},
    'MADURAI': {'N': 160, 'P': 14, 'K': 390, 'pH': 7.5}, 'THE NILGIRIS': {'N': 350, 'P': 40, 'K': 200, 'pH': 4.5}
}

def get_live_weather(district_name):
    """Fetches live weather using API"""
    coords = DISTRICT_COORDS.get(district_name.upper().strip())
    if not coords:
        return None, "District coordinates not found."
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={coords['lat']}&lon={coords['lon']}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            return temp, desc
        else:
            return None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, str(e)

# --- MAIN APP LOGIC ---
df = load_data()

if df is not None:
    # Load Models
    regressor, classifier, le_dist, le_seas, le_crop = load_or_train_models(df)

    st.sidebar.title("")
    page = st.sidebar.radio("Go to", ["Crop Recommender", "Yield Predictor", "Geo Analysis", "Soil Doctor"])

    # ==========================
    # 1. CROP RECOMMENDER
    # ==========================
    if page == "Crop Recommender":
        st.title("🌱 Crop Recommendation")
        st.markdown("AI-driven analysis to suggest the most suitable crop for your soil conditions.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📍 Location & Season")
            districts = sorted(df['District_Name'].unique().astype(str))
            sel_dist = st.selectbox("District", districts)
            seasons = sorted(df['Season'].unique().astype(str))
            sel_season = st.selectbox("Season", seasons)
            
            # Use dictionary defaults if available, else generic
            defaults = SOIL_DEFAULTS.get(sel_dist, {'N': 220, 'P': 18, 'K': 300, 'pH': 7.2})
            
            st.subheader("🌦️ Environment")
            rain = st.slider("Expected Rainfall (mm)", 0.0, 3000.0, 1200.0)
            temp = st.slider("Avg Temperature (°C)", 10.0, 45.0, 28.0)

        with col2:
            st.subheader("🧪 Soil Profile (kg/ha)")
            n = st.number_input("Nitrogen (N)", value=defaults['N'])
            p = st.number_input("Phosphorus (P)", value=defaults['P'])
            k = st.number_input("Potassium (K)", value=defaults['K'])
            ph = st.number_input("Soil pH", value=defaults['pH'], step=0.1)

        st.markdown("---")
        if st.button("Recommend Crop"):
            try:
                # Prepare Input
                d_enc = le_dist.transform([sel_dist])[0]
                s_enc = le_seas.transform([sel_season])[0]
                
                input_vec = pd.DataFrame([[d_enc, s_enc, n, p, k, ph, rain, temp]], 
                                         columns=['District_Encoded', 'Season_Encoded', 'N', 'P', 'K', 'pH', 'Rainfall', 'Temperature'])
                
                # Predict Probabilities
                probs = classifier.predict_proba(input_vec)[0]
                
                # Get Top 3
                top3_idx = np.argsort(probs)[-3:][::-1]
                top3_crops = le_crop.inverse_transform(top3_idx)
                top3_probs = probs[top3_idx]
                
                st.success(f"🌟 **Top Recommendation:** {top3_crops[0]} ({top3_probs[0]*100:.1f}% Match)")
                
                # Bar Chart
                chart_data = pd.DataFrame({'Crop': top3_crops, 'Confidence': top3_probs})
                fig = px.bar(chart_data, x='Crop', y='Confidence', 
                             color='Confidence',
                             color_continuous_scale='Greens',
                             title="Crop Suitability Index")
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#E0E0E0'
                )
                st.plotly_chart(fig, width="stretch")
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")

    # ==========================
    # 2. YIELD PREDICTOR
    # ==========================
    elif page == "Yield Predictor":
        st.title("🤖 Yield Prediction")
        
        weather_temp = 28.0
        weather_status = None
        
        col_main, col_soil = st.columns([1, 1])
        
        with col_main:
            st.subheader("🌾 Crop Details")
            districts = sorted(df['District_Name'].unique().astype(str))
            sel_dist = st.selectbox("District", districts)
            
            # API Weather Button
            if st.button("☁️ Fetch Live Weather"):
                if OPENWEATHER_API_KEY != "YOUR_API_KEY_HERE":
                    live_temp, live_desc = get_live_weather(sel_dist)
                    if live_temp:
                        weather_temp = live_temp
                        weather_status = f"✅ Live Data: {live_temp}°C, {live_desc.title()}"
                        st.success(f"Fetched: {live_temp}°C ({live_desc})")
                    else:
                        st.error(f"Failed to fetch weather: {live_desc}")
                else:
                    st.warning("Please update the API Key in the code to use Live Weather.")

            seasons = sorted(df['Season'].unique().astype(str))
            sel_season = st.selectbox("Season", seasons)
            crops = sorted(df['Crop'].unique().astype(str))
            sel_crop = st.selectbox("Crop", crops)
            area = st.number_input("Area (Hectares)", value=1.0)

        with col_soil:
            st.subheader("🌍 Conditions")
            defaults = SOIL_DEFAULTS.get(sel_dist, {'N': 220, 'P': 18, 'K': 300, 'pH': 7.2})
            
            temp = st.slider("Temperature (°C)", 10.0, 45.0, value=float(weather_temp))
            if weather_status: st.caption(weather_status)
            rain = st.slider("Rainfall (mm)", 0.0, 3000.0, 1200.0)
            
            c1, c2 = st.columns(2)
            n_val = c1.number_input("N (kg/ha)", value=defaults['N'])
            p_val = c2.number_input("P (kg/ha)", value=defaults['P'])
            k_val = c1.number_input("K (kg/ha)", value=defaults['K'])
            ph_val = c2.number_input("pH", value=defaults['pH'])

        st.markdown("---")
        if st.button("Calculate Yield"):
            try:
                d_c = le_dist.transform([sel_dist])[0]
                s_c = le_seas.transform([sel_season])[0]
                c_c = le_crop.transform([sel_crop])[0]
                
                # IMPORTANT: Feature order must match training
                # ['District_Encoded', 'Season_Encoded', 'Crop_Encoded', 'Area', 'Rainfall', 'Temperature', 'N', 'P', 'K', 'pH']
                input_data = pd.DataFrame([[d_c, s_c, c_c, area, rain, temp, n_val, p_val, k_val, ph_val]],
                                          columns=regressor.feature_names_in_)
                
                pred_yield = regressor.predict(input_data)[0]
                total_prod = pred_yield * area
                
                m1, m2 = st.columns(2)
                m1.metric("Predicted Yield", f"{pred_yield:.2f} Tonnes/Ha")
                m2.metric("Total Production", f"{total_prod:.2f} Tonnes")
                
            except Exception as e:
                st.error(f"Error: {e}")

    # ==========================
    # 3. GEO ANALYSIS
    # ==========================
    elif page == "Geo Analysis":
        st.title("🗺️ Spatial Analysis")
        
        # Calculate avg yield per district
        map_data = df.groupby('District_Name')['Yield'].mean().reset_index()
        
        # Add lat/lon from dictionary
        def get_coords(n):
            return pd.Series([DISTRICT_COORDS.get(n.upper().strip(), {}).get('lat'), 
                              DISTRICT_COORDS.get(n.upper().strip(), {}).get('lon')])
        
        map_data[['lat', 'lon']] = map_data['District_Name'].apply(get_coords)
        map_data = map_data.dropna()
        
        map_data['Display_Yield'] = map_data['Yield'].apply(lambda x: f"{x:.2f} T/Ha")

        fig_map = px.scatter_mapbox(
            map_data, lat="lat", lon="lon", size="Yield", color="Yield",
            hover_name="District_Name", 
            hover_data={'lat':False, 'lon':False, 'Yield':False, 'Display_Yield':True},
            color_continuous_scale='Greens', 
            size_max=40, zoom=6,
            mapbox_style="carto-darkmatter", # Dark map style
            height=600, 
            title="Avg Yield Potential (Tonnes/Ha)"
        )
        fig_map.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#E0E0E0'
        )
        st.plotly_chart(fig_map, width="stretch")

    # ==========================
    # 4. SOIL DOCTOR
    # ==========================
    elif page == "Soil Doctor":
        st.title("🧪 Soil Health Diagnostics")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Input Readings (kg/ha)")
            n = st.number_input("Nitrogen (N)", 0, 500, 150)
            p = st.number_input("Phosphorus (P)", 0, 200, 20)
            k = st.number_input("Potassium (K)", 0, 500, 150)
            ph = st.number_input("pH Level", 0.0, 14.0, 6.5)
        
        with c2:
            st.subheader("Health Status")
            # Gauge Chart for Nitrogen
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = n, 
                title = {'text': "Nitrogen (kg/ha)"},
                gauge = {
                    'axis': {'range': [0, 500]}, 
                    'bar': {'color': "#4CAF50"}, # Brighter green for gauge
                    'steps': [
                        {'range': [0, 200], 'color': "#FF8A80"}, # Reddish for low
                        {'range': [200, 500], 'color': "#66BB6A"} # Greenish for good
                    ],
                    'bgcolor': '#333333',
                    'bordercolor': '#333333'
                },
                number={'font': {'color': '#FFFFFF'}} 
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                font={'color': "#E0E0E0"},
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width="stretch")
            
            st.info("💡 **AI Recommendation:**")
            
            # Rule-based recommendations
            if n < 200: st.write("- **Low Nitrogen:** Apply Urea top dressing.")
            else: st.success("- **Nitrogen** levels are optimal.")
            
            if p < 30: st.write("- **Low Phosphorus:** Use DAP (Di-ammonium Phosphate).")
            if k < 150: st.write("- **Low Potassium:** Apply MOP (Muriate of Potash).")
            
            if ph < 6.0: st.warning("- **Acidic Soil:** Treat with Lime.")
            elif ph > 7.5: st.warning("- **Alkaline Soil:** Treat with Gypsum.")
            else: st.success("- **Soil pH** is neutral and healthy.")

else:
    st.error("Dataset not found. Please place 'crop_yield_final_dataset.csv' in the directory.")