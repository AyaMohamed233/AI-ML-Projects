 🛒 Walmart Sales Forecasting

This project focuses on forecasting weekly sales for Walmart stores using historical sales data, store metadata, and economic indicators. The goal is to predict sales more accurately to improve supply chain and inventory decisions.

## 📌 Problem Statement

Retail companies like Walmart require accurate sales forecasting to manage operations, stock, and promotional strategies. Given historical sales data across multiple departments and stores, our goal is to build regression models to forecast future weekly sales.

---

📂 Dataset

The dataset is sourced from [Kaggle - Walmart Recruiting: Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data) and contains the following CSV files:

- `train.csv`: Historical weekly sales data
- `test.csv`: Test set to predict sales
- `features.csv`: Additional data like temperature, fuel price, CPI, etc.
- `stores.csv`: Store-level information

---

## ⚙️ Tools & Libraries Used

- **Python** 3.11
- **Pandas**, **NumPy** for data processing
- **Matplotlib**, **Seaborn** for visualization
- **Scikit-learn** for machine learning models
- **Random Forest Regressor**, **Gradient Boosting Regressor**

---

## 📊 Exploratory Data Analysis (EDA)

We explored several aspects of the data, including:

- Total weekly sales over time by store type
- Sales distribution by month
- Comparison of holiday vs. non-holiday weeks
- Top-performing departments
- Correlation between sales and economic indicators (CPI, unemployment)
- Seasonal patterns and store size impacts

Visualizations were used to identify patterns, trends, and relationships in the dataset.

---

## 🧠 Feature Engineering

Created new features to improve model performance:
- Temporal: `Year`, `Month`, `Week`, `Quarter`, `DayOfYear`
- Event-based: `IsHoliday`, `IsChristmasWeek`, `IsBackToSchool`, `IsSummer`
- Combined features: `Size_Temperature`, `CPI_Unemployment`
- Encoded composite IDs: `Store_Dept_Encoded`

---

## 🧪 Model Training & Evaluation

Two regression models were trained and evaluated:

| Model             | RMSE     | MAE     | R² Score | CV R² Score |
|------------------|----------|---------|----------|-------------|
| Random Forest     | 8074.99  | 4223.04 | 0.8750   | 0.8849 ± 0.0049 |
| Gradient Boosting | *In progress* | *TBD*  | *TBD*   | *TBD*       |

**Best Model:** Selected based on the highest R² score on validation data.

---

## 📈 Feature Importance

The most influential features in the model were:

1. Store_Dept_Encoded  
2. Size  
3. Month  
4. Dept  
5. Store  
6. CPI_Unemployment  
7. Fuel_Price  
8. Week  
9. Size_Temperature  
10. Unemployment  

---

## 🔮 Prediction & Submission

- Final predictions were made on the `test.csv` dataset.
- Results were saved to `enhanced_submission.csv` with the required `Id` and `Weekly_Sales` format.

---

## 📦 Project Structure
Sales Forecasting/
│
├── train.csv
├── test.csv
├── features.csv
├── stores.csv
├── main.py
├── enhanced_submission.csv
└── README.md
