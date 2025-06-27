# 🏪 Rossmann Sales Forecasting App 📈

A machine learning project that forecasts daily sales for Rossmann retail stores using real-world data. This project uses XGBoost and time-series feature engineering, and presents results in an interactive **Streamlit dashboard**.

---

## 📌 Project Highlights

- 🔍 **Exploratory Data Analysis**: Promo, Holidays, and Competition effects analyzed.
- 🔧 **Feature Engineering**: Created time-based, promo-based, and lag-based features.
- 🧠 **Modeling**: Trained XGBoost model to predict sales using engineered features.
- 📊 **Streamlit Dashboard**: Choose a store and date to view predicted sales with historical trends.
- 📁 **Production-Ready**: Model and data preprocessing saved for easy deployment.


---

## 🧠 ML Techniques Used

- **Time-based features**: Year, Month, Day, DayOfWeek, Quarter, WeekOfYear
- **Lag features**: Previous day and week sales, 7-day rolling average
- **Promotion features**: Promo flag, IsPromoMonth, Promo2 duration
- **Competition features**: Competition distance and open duration
- **Model**: XGBoost Regressor (tuned for performance)

---

## 📁 Folder Structure

sales forecasting for Retail Chain/
│
├── app.py # 📱 Streamlit dashboard
├── model/
│ └── xgb_model.pkl # 💾 Trained model
├── data/
│ └── processed_df.csv # ✅ Cleaned + feature engineered dataset
├── requirements.txt # 📦 Python dependencies
├── README.md # 📘 This file



📚 Dataset Info
Dataset: Rossmann Store Sales (Kaggle)

Duration: Jan 2013 to July 2015

Stores: 1,111 locations

Provided: Store meta data, promotions, competition info, and daily sales

✅ Project Learnings
Handling real-world time-series data

Generating lag and rolling features for machine learning

Visualizing promotions, holidays, and competition impact

Deploying an ML model with Streamlit