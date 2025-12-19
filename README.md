# ✈️ Airline Passenger Satisfaction Prediction

This project implements a **machine learning–based predictive analytics system** to estimate **flight passenger satisfaction scores** using historical airline passenger and flight data.

The project follows a complete **end-to-end machine learning workflow**, including data preprocessing, feature engineering, model training, evaluation, and model saving.

---

## 📌 Problem Statement

Passenger satisfaction in the airline industry depends on multiple factors such as delays, pricing, distance traveled, and booking behavior. Traditional analysis methods fail to capture complex relationships between these variables.

**Objective:**  
To build a regression-based machine learning model that predicts **Flight Satisfaction Score** accurately.

---

## 📊 Dataset

- **Name:** Synthetic Airline Passenger and Flight Data  
- **Source:** Kaggle  
- **Type:** Synthetic, structured tabular data  
- **Records:** Thousands of passenger–flight entries  

### Target Variable
- `Flight_Satisfaction_Score`

---

## ⚙️ Technologies Used

- **Programming Language:** Python  
- **Libraries:**
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - Joblib

---

## 🧹 Data Preprocessing

- Removed duplicate records  
- Handled missing values (median for numeric, mode for categorical)  
- Encoded categorical variables using Label Encoding  
- Feature engineering:
  - Delay Severity
  - Price per Mile
  - Booking Type  
- Train-test split (80/20)

---

## 📈 Exploratory Data Analysis (EDA)

- Distribution of satisfaction scores  
- Impact of delay duration on satisfaction  
- Pricing efficiency vs passenger experience  
- Booking behavior analysis  

---

## 🤖 Machine Learning Models

The following regression models were trained and evaluated:

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  

---

## 🏆 Model Evaluation

Models were evaluated using:

- R² Score  
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  

✅ **Random Forest Regressor** performed best and was selected as the final model.

---

## 💾 Model Saving

The best-performing model was saved using Joblib:

