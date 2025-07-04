## Bank Customer Churn Prevention System

An AI-powered solution that predicts customer churn risk and recommends optimal retention strategies to maximize ROI while respecting budget constraints.

## Key Features

- **Predictive Analytics**: CatBoost model trained on historical banking data
- **Dynamic Optimization**: Real-time strategy allocation using linear programming
- **Business-Friendly Dashboard**: Streamlit interface with interactive controls
- **ROI-Focused**: Calculates Customer Lifetime Value (CLV) for each intervention
- **Explainable AI**: SHAP values show why customers are at risk

## Tech Stack

- **Machine Learning**: Python, Catboost, Scikit-learn, SHAP
- **Optimization**: SciPy Linear Programming
- **Visualization**: Plotly, Matplotlib
- **Web App**: Streamlit
- **Data Processing**: Pandas, NumPy


## Data Preparation

Place your bank customer data in `data/` with these required columns:
- `CustomerId`
- `Balance`
- `Tenure` 
- `CreditScore`
- `NumOfProducts`
- `IsActiveMember`
- `Exited` (target variable)


## Model Training

The trained model is saved to `models/catboost.pkl`

## Running the App

Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Project Structure

```
bank-churn-optimizer/
├── app.py                 # Streamlit application
├── intervention_optimizer.py  # ROI optimization engine
├── data/ # Original datasets
│    
│   └── predictions/       # data after prediction
├── models/                # Trained ML models
├── requirements.txt       # Python dependencies
└── README.md
```

## Configuration

Customize these parameters in `app.py`:
```python
# Strategy parameters
STRATEGIES = {
    "Wealth Manager Call": (75, 0.50, 3),  # (cost, risk_reduction, max_usage)
    "Credit Limit Increase": (40, 0.35, 5),
    # ...
}

# CLV calculation
MONTHLY_YIELD = 0.0045  # 0.45% of balance
MAX_CLV_YEARS = 10      # Cap CLV calculation
```

## Sample Output
https://github.com/Pratik-Mali-11/Data-Science-Project-Series/blob/5340be217e7999430dd59d6f172ec858a32e16fb/Bank%20customer%20Churn%20Prediction/streamlit_output.png

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Pratik Mali - malipratik097@gmail.com
Project Link: [https://github.com/yourusername/bank-churn-optimizer](https://github.com/yourusername/bank-churn-optimizer)
```

