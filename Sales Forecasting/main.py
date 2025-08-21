import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
features = pd.read_csv("features.csv")
stores = pd.read_csv("stores.csv")

# Convert dates
train["Date"] = pd.to_datetime(train["Date"])
features["Date"] = pd.to_datetime(features["Date"])
test["Date"] = pd.to_datetime(test["Date"])

# Remove duplicate IsHoliday from train (it's in features too)
train = train.drop(columns=['IsHoliday'])

# Merge datasets
train_merged = pd.merge(train, features, how='left', on=['Store', 'Date'])
train_merged = pd.merge(train_merged, stores, how='left', on='Store')

print(f"Training data shape: {train_merged.shape}")
print(f"Missing values in training data:\n{train_merged.isnull().sum()}")

def create_features(df):

    df = df.copy()

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Quarter'] = df['Date'].dt.quarter
  
    df['IsChristmasWeek'] = ((df['Month'] == 12) & (df['Week'] >= 51)).astype(int)
    df['IsSummer'] = df['Month'].isin([6, 7, 8]).astype(int)
    df['IsBackToSchool'] = ((df['Month'] == 8) | ((df['Month'] == 9) & (df['Week'] <= 2))).astype(int)

    df['Store_Dept'] = df['Store'].astype(str) + '_' + df['Dept'].astype(str)
  
    le_store_dept = LabelEncoder()
    df['Store_Dept_Encoded'] = le_store_dept.fit_transform(df['Store_Dept'])

    df['Type'] = df['Type'].map({'A': 1, 'B': 2, 'C': 3})
    df['IsHoliday'] = df['IsHoliday'].astype(int)
    
    df['Size_Temperature'] = df['Size'] * df['Temperature']
    df['CPI_Unemployment'] = df['CPI'] * df['Unemployment']
    
    return df

train_merged = create_features(train_merged)

plt.style.use('default')
sns.set_palette("husl")

plt.figure(figsize=(16, 8))
for store_type in train_merged['Type'].unique():
    if not pd.isna(store_type):
        type_data = train_merged[train_merged['Type'] == store_type]
        weekly_sales = type_data.groupby("Date")["Weekly_Sales"].sum()
        plt.plot(weekly_sales.index, weekly_sales.values, label=f'Store Type {int(store_type)}', 
                linewidth=2, alpha=0.8)
plt.title("Total Weekly Sales Over Time by Store Type", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Total Weekly Sales ($)", fontsize=14)
plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()

# 2. Sales distribution by month
plt.figure(figsize=(14, 8))
sns.boxplot(data=train_merged, x="Month", y="Weekly_Sales", palette='Set2')
plt.title("Sales Distribution by Month", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Weekly Sales ($)", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Holiday vs non-holiday sales
plt.figure(figsize=(10, 8))
holiday_sales = train_merged.groupby('IsHoliday')['Weekly_Sales'].agg(['mean', 'std']).reset_index()
holiday_sales['IsHoliday_Label'] = holiday_sales['IsHoliday'].map({0: 'Regular Days', 1: 'Holiday Weeks'})
bars = plt.bar(holiday_sales['IsHoliday_Label'], holiday_sales['mean'], 
               color=['skyblue', 'coral'], alpha=0.8, edgecolor='black', linewidth=1)
plt.errorbar(holiday_sales['IsHoliday_Label'], holiday_sales['mean'], 
             yerr=holiday_sales['std'], fmt='none', color='black', capsize=5)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + holiday_sales.iloc[i]['std'],
             f'${height:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title("Average Weekly Sales: Regular Days vs Holiday Weeks", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Day Type", fontsize=14)
plt.ylabel("Average Weekly Sales ($)", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 4. Top departments by sales
plt.figure(figsize=(14, 8))
top_depts = train_merged.groupby('Dept')['Weekly_Sales'].mean().sort_values(ascending=False).head(12)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_depts)))
bars = plt.bar(range(len(top_depts)), top_depts.values, color=colors, alpha=0.8, edgecolor='black')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("Top 12 Departments by Average Weekly Sales", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Department (Ranked by Sales)", fontsize=14)
plt.ylabel("Average Weekly Sales ($)", fontsize=14)
plt.xticks(range(len(top_depts)), [f'Dept {int(d)}' for d in top_depts.index], rotation=45)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 5. Sales vs Temperature (with trend line)
plt.figure(figsize=(12, 8))
# Sample data to avoid overplotting
sample_data = train_merged.sample(n=min(10000, len(train_merged)), random_state=42)
plt.scatter(sample_data['Temperature'], sample_data['Weekly_Sales'], 
           alpha=0.4, color='steelblue', s=20)
# Add trend line
z = np.polyfit(sample_data['Temperature'], sample_data['Weekly_Sales'], 1)
p = np.poly1d(z)
plt.plot(sample_data['Temperature'].sort_values(), p(sample_data['Temperature'].sort_values()), 
         "r--", alpha=0.8, linewidth=2, label=f'Trend Line')
plt.title("Weekly Sales vs Temperature", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Temperature (°F)", fontsize=14)
plt.ylabel("Weekly Sales ($)", fontsize=14)
plt.legend(fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Correlation heatmap (larger and cleaner)
plt.figure(figsize=(12, 10))
numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'Type']
corr_matrix = train_merged[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix))  # Show only lower triangle
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, 
           square=True, fmt='.3f', cbar_kws={"shrink": .8}, 
           annot_kws={'size': 12, 'weight': 'bold'})
plt.title("Feature Correlation Matrix", fontsize=16, fontweight='bold', pad=20)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()

# 7. Sales by store size (improved)
plt.figure(figsize=(12, 8))
train_merged['Size_Bin'] = pd.qcut(train_merged['Size'], 5, labels=['Extra Small', 'Small', 'Medium', 'Large', 'Extra Large'])
size_sales = train_merged.groupby('Size_Bin')['Weekly_Sales'].agg(['mean', 'std']).reset_index()
bars = plt.bar(size_sales['Size_Bin'], size_sales['mean'], 
              color='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=2)
plt.errorbar(size_sales['Size_Bin'], size_sales['mean'], 
             yerr=size_sales['std'], fmt='none', color='black', capsize=5)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + size_sales.iloc[i]['std'],
             f'${height:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title("Average Weekly Sales by Store Size Category", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Store Size Category", fontsize=14)
plt.ylabel("Average Weekly Sales ($)", fontsize=14)
plt.xticks(rotation=45)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 8. Seasonal trends (enhanced)
plt.figure(figsize=(14, 8))
seasonal_sales = train_merged.groupby('Month')['Weekly_Sales'].agg(['mean', 'std']).reset_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.plot(seasonal_sales['Month'], seasonal_sales['mean'], 
         marker='o', linewidth=3, markersize=8, color='darkred', alpha=0.8)
plt.fill_between(seasonal_sales['Month'], 
                seasonal_sales['mean'] - seasonal_sales['std'],
                seasonal_sales['mean'] + seasonal_sales['std'], 
                alpha=0.2, color='red')

# Highlight peak months
peak_months = seasonal_sales.nlargest(3, 'mean')['Month'].values
for month in peak_months:
    plt.axvline(x=month, color='gold', linestyle='--', alpha=0.7, linewidth=2)

plt.title("Seasonal Sales Trends with Variability", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Average Weekly Sales ($)", fontsize=14)
plt.xticks(range(1, 13), month_names)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True, alpha=0.3)
plt.legend(['Average Sales', 'Sales Range (±1 std)', 'Peak Months'], 
          fontsize=12, loc='upper right')
plt.tight_layout()
plt.show()

# Prepare features for modeling
features_used = [
    'Store', 'Dept', 'Temperature', 'Fuel_Price', 'CPI',
    'Unemployment', 'Size', 'Type', 'IsHoliday', 'Year', 'Month', 'Week',
    'DayOfYear', 'Quarter', 'IsChristmasWeek', 'IsSummer', 'IsBackToSchool',
    'Store_Dept_Encoded', 'Size_Temperature', 'CPI_Unemployment'
]

# Handle missing values more carefully
X = train_merged[features_used].copy()
y = train_merged['Weekly_Sales'].copy()

# Fill missing values with median for numerical features
for col in X.columns:
    if X[col].dtype in ['float64', 'int64']:
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Try multiple models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
,
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
}

model_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_val)
    
    # Metrics
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    model_results[name] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"{name} Results:")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  MAE: {mae:,.2f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Select best model
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
best_model = model_results[best_model_name]['model']
print(f"\nBest model: {best_model_name}")

# Feature importance analysis with better visualization
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': features_used,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:20s} - {row['importance']:.4f}")
    
    plt.figure(figsize=(14, 10))
    top_features = feature_importance.head(15)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['importance'], 
                    color=colors, alpha=0.8, edgecolor='black')
    
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=12)
    plt.xlabel('Feature Importance', fontsize=14, fontweight='bold')
    plt.title(f'Top 15 Feature Importance - {best_model_name}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add importance values as text
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

# Enhanced visualization of predictions with better readability
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
best_pred = model_results[best_model_name]['predictions']

# 1. Actual vs Predicted scatter plot
axes[0, 0].scatter(y_val, best_pred, alpha=0.6, color='steelblue', s=20)
# Perfect prediction line
min_val, max_val = min(y_val.min(), best_pred.min()), max(y_val.max(), best_pred.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Sales ($)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Predicted Sales ($)', fontsize=12, fontweight='bold')
axes[0, 0].set_title(f'Actual vs Predicted - {best_model_name}', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='both', which='major', labelsize=10)

# Add R² score as text
r2_score_val = model_results[best_model_name]['r2']
axes[0, 0].text(0.05, 0.95, f'R² = {r2_score_val:.4f}', 
                transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 2. Residual plot
residuals = y_val - best_pred
axes[0, 1].scatter(best_pred, residuals, alpha=0.6, color='coral', s=20)
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Sales ($)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Residuals ($)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='both', which='major', labelsize=10)

# 3. Distribution of residuals
axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1, 0].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: ${residuals.mean():,.0f}')
axes[1, 0].axvline(residuals.median(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Median: ${residuals.median():,.0f}')
axes[1, 0].set_xlabel('Residuals ($)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='both', which='major', labelsize=10)

# 4. Sample predictions comparison
sample_size = min(100, len(y_val))
sample_idx = np.random.choice(len(y_val), sample_size, replace=False)
sample_actual = y_val.iloc[sample_idx].values
sample_predicted = best_pred[sample_idx]

x_pos = np.arange(sample_size)
axes[1, 1].plot(x_pos, sample_actual, 'o-', label='Actual', alpha=0.8, 
                color='darkblue', linewidth=2, markersize=4)
axes[1, 1].plot(x_pos, sample_predicted, 's-', label='Predicted', alpha=0.8, 
                color='darkred', linewidth=2, markersize=4)
axes[1, 1].set_xlabel('Sample Index', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
axes[1, 1].set_title(f'Sample Predictions Comparison (n={sample_size})', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout(pad=3.0)
plt.show()

# Model performance summary table
print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"{'Metric':<20} {'Value':<15} {'Description'}")
print("-"*60)
print(f"{'RMSE':<20} ${model_results[best_model_name]['rmse']:>12,.0f} {'Root Mean Squared Error'}")
print(f"{'MAE':<20} ${model_results[best_model_name]['mae']:>12,.0f} {'Mean Absolute Error'}")
print(f"{'R² Score':<20} {model_results[best_model_name]['r2']:>14.4f} {'Coefficient of Determination'}")
print(f"{'CV R² Mean':<20} {model_results[best_model_name]['cv_mean']:>14.4f} {'Cross-Validation R² Mean'}")
print(f"{'CV R² Std':<20} {model_results[best_model_name]['cv_std']:>14.4f} {'Cross-Validation R² Std'}")
print("-"*60)

# Prepare test data
print("\nPreparing test predictions...")
test_merged = pd.merge(test, features, how='left', on=['Store', 'Date'])
test_merged = pd.merge(test_merged, stores, how='left', on='Store')

# Ensure IsHoliday is in test_merged
if 'IsHoliday' not in test_merged.columns:
    test_merged = pd.merge(test_merged, test[['Store', 'Date', 'IsHoliday']], how='left', on=['Store', 'Date'])

# Apply same feature engineering to test data
test_merged = create_features(test_merged)

# Prepare test features
X_test = test_merged[features_used].copy()

# Handle missing values in test set
for col in X_test.columns:
    if X_test[col].dtype in ['float64', 'int64']:
        X_test[col] = X_test[col].fillna(X[col].median())  # Use training median
    else:
        X_test[col] = X_test[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)

print(f"Test features shape: {X_test.shape}")
print(f"Missing values in test set: {X_test.isnull().sum().sum()}")

# ↓↓↓ Add this ↓↓↓
X_test = X_test.sample(n=200000, random_state=42)
print("⚠️ Working with a sample of 200,000 rows from test set due to memory limits.")


# Generate predictions
y_test_pred = best_model.predict(X_test)

# Create submission file
test['Weekly_Sales'] = y_test_pred
test['Id'] = test['Store'].astype(str) + "_" + test['Dept'].astype(str) + "_" + test['Date'].dt.strftime('%Y-%m-%d')

submission = test[['Id', 'Weekly_Sales']]
submission.to_csv("enhanced_submission.csv", index=False)

print(f"\n✅ Saved: enhanced_submission.csv")
print(f"Final model: {best_model_name}")
print(f"Final R² Score: {model_results[best_model_name]['r2']:.4f}")
print(f"Final RMSE: {model_results[best_model_name]['rmse']:,.2f}")