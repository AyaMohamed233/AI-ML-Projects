import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv(r"loan_approval_dataset.csv")
data.columns = data.columns.str.strip()
data.drop(['loan_id'],axis=1,inplace=True)
data['education'] = data['education'].str.strip()
data['self_employed'] = data['self_employed'].str.strip()
data['loan_status'] = data['loan_status'].str.strip()
le=LabelEncoder()
data['education'] = le.fit_transform(data['education'])
data['self_employed'] = le.fit_transform(data['self_employed'])
data['loan_status'] = le.fit_transform(data['loan_status'])

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers)

for col in numeric_cols:
    num_outliers = detect_outliers_iqr(data, col)
    print(f'{col}: {num_outliers} outliers')

df_cleaned = data.copy()

for col in ['residential_assets_value', 'commercial_assets_value', 'bank_asset_value']:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_cleaned = df_cleaned[(df_cleaned[col] >= lower) & (df_cleaned[col] <= upper)]
df_cleaned.to_csv("loan_data.csv", index=False)
