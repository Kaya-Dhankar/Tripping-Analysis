# Tripping-Analysis

This project is a **Streamlit-based web application** for analyzing **tripping and synchronization events** in power plants (CSPGCL).  
It combines **MongoDB** for storing event logs, **machine learning** for predicting tripping reasons, and a **user-friendly UI** for engineers/operators.

---

## 🚀 Features
- 🏭 **Station & Unit Selection**: Choose power station and unit dynamically from MongoDB.  
- 🕒 **Event Logging**: Input **Last Tripping, Lit-up, and Synchronization times**.  
- 🔍 **Database Search**: Finds exact matches in MongoDB tripping logs.  
- 🤖 **ML Prediction**: If no match exists, uses a trained model (`tripping_model.pkl`) to predict the most likely reason.  
- 🎨 **Beautiful UI**: Custom background, styled result cards, and clear color-coded status messages.  
- 📦 **MongoDB Integration**: Stores and retrieves structured logs.  

---

## 🗂️ Project Structure
Tripping_Analysis/
│
├── data.py # Main Streamlit application
├── requirements.txt # Dependencies
├── tripping_model.pkl # Pickled ML model + encoders
├── tripping_predictor.pkl # Extra predictor file
├── thermal power plant.jpg # Background image
└── README.md # Project documentation
