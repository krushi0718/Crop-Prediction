🌾 TN Agri-Intelligence Hub

An AI-Powered Agriculture Decision Support System for Tamil Nadu

This project is a comprehensive data science and web application solution designed to assist farmers and agricultural policymakers in Tamil Nadu. It leverages historical yield data, real-time weather APIs, and scientific soil profiles to provide actionable insights.

🌟 Features

1. 🌱 Smart Crop Recommender  
Uses a Machine Learning Classifier to analyze soil composition (NPK), pH levels, and expected weather.  
Recommends the top 3 most suitable crops with confidence scores.

2. 🤖 Yield Predictor  
Uses a Random Forest Regressor trained on historical Tamil Nadu agriculture data enriched with Open-Meteo weather archives.  
Predicts expected yield (Tonnes/Hectare) and total production.

3. 🗺️ Geo-Spatial Analysis  
Aggregates historical performance data by district.  
Displays an interactive dark-mode map of Tamil Nadu.

4. 🧪 Soil Doctor  
Compares user soil inputs against agronomic standards.  
Provides fertilizer recommendations and soil health visualization.

📂 Project Structure

- `app.py` – Streamlit web application  
- `cropred.ipynb` – Data science & model training pipeline  
- `requirements.txt` – Dependencies  
- `*.pkl` – Saved*
