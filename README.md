# Reliance India Stock Price Prediction Model 📈

This repository contains a **Streamlit web application** that forecasts the stock price of **Reliance Industries Ltd. (RELI)**.  
It uses **10 years of historical stock price data** and applies both **LightGBM** and **Linear Regression** models for prediction and comparison.

---

## 🚀 Features
- 📊 **Data Preprocessing**  
  Cleans raw stock price data, converts dates, and calculates 5-day & 10-day moving averages.
- 🤖 **Model Training**  
  - **LightGBM Regressor**  
  - **Linear Regression**  
  Evaluated using MAE and R² metrics.
- 📉 **Actual vs Predicted Chart**  
  Interactive Plotly chart comparing real vs model-predicted prices.
- 🔮 **Forecasting**  
  Forecasts Reliance stock price for **August 2025 (31 days)**.
- 🌐 **Streamlit UI**  
  Clean and interactive dashboard with sidebar project info.

---

## 📂 Repository Structure
```
.
├── reli_final.py                # Main Streamlit app
├── RELI Historical Data.csv     # Historical stock data (10 years)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/manasmalviya/Reliance-India-Stock-Prediction-Model.git
   cd Reliance-India-Stock-Prediction-Model
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run reli_final.py
   ```

---

## 📦 Requirements
Dependencies are listed in `requirements.txt`:

```
streamlit
pandas
numpy
lightgbm
plotly
scikit-learn
```

---

## ☁️ Deployment on Streamlit Cloud
1. Push this repo to GitHub.  
2. Go to [Streamlit Cloud](https://share.streamlit.io).  
3. Sign in with your GitHub account.  
4. Click **New App** → select this repo.  
5. Set:
   - **Branch**: `main`
   - **File path**: `reli_final.py`  
6. Deploy 🎉

---

## 📊 Example Output
- **Model Performance Metrics** (MAE, R²)
- **Interactive Actual vs Predicted Plot**
- **Forecasted Prices for August 2025**
- Expandable forecast table

---

## 📝 Notes
- Ensure `RELI Historical Data.csv` is in the same directory as `reli_final.py`.
- If you want to try another stock, replace the CSV with its historical data in the same format.

---

## 👨‍💻 Author
**Manas Malviya**  
📌 [GitHub Profile](https://github.com/manasmalviya)

---
