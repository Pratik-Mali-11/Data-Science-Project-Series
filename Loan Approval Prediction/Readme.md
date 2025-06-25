Loan Approval Prediction Dashboard
This project is a complete Loan Approval Prediction System built with:
•	Machine Learning (Random Forest Classifier)
•	Feature Engineering & EDA
•	 Streamlit Dashboards for real-time prediction
•	Upload support with loan ID selection
________________________________________
About the Project
This machine learning project aims to predict whether a loan application will be approved or not, using demographic and financial features like:
•	Applicant Income
•	Credit History
•	Property Area
•	Education, Gender, Marital Status, etc.
________________________________________
Features
 Manual input dashboard (app.py),
 Upload CSV + Select Loan ID dashboard (app2.py),
 Handles feature engineering (log transform, income binning),
 Trained Random Forest classifier,
 Model accuracy and performance included,
 Supports future improvements with SHAP/LIME or batch predictions
________________________________________
How to Run
1.	Install dependencies:
pip install -r requirements.txt
2.	To launch manual input dashboard:
streamlit run app.py
3.	To launch file upload + ID selection dashboard:
streamlit run app2.py
________________________________________
app.py vs app2.py
File	Description
app.py	Manual form-based loan prediction. You fill values like income, credit, etc.
app2.py	Upload a dataset (CSV), select a Loan_ID, and predict approval result.
________________________________________
 Model Training
The model was trained on the Loan Prediction dataset (from Analytics Vidhya) using:
•	Feature Engineering: total income, log(loan), income categories
•	Encoding: label + one-hot
•	Model: Random Forest Classifier
•	Evaluation: Accuracy, Precision, Recall, F1-score
________________________________________
 Sample Dataset Format
Make sure uploaded dataset for app2.py contains the following columns:
Loan_ID, Gender, Married, Education, Self_Employed, Dependents,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area
________________________________________
Requirements
•	Python 3.8+
•	Streamlit
•	pandas, numpy, joblib, scikit-learn
See requirements.txt for full list.
________________________________________
Credits
Dataset: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
Model & Dashboard: Built by Pratik Mali
________________________________________
