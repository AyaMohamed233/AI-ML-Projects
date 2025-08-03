import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,DBSCAN
df=pd.read_csv(r'D:\ai projects\Customer Segmentation\Mall_Customers.csv')
data = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set1',
    s=100
)
plt.title('Customer Segments based on Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(scaled_data)
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['DBSCAN_Cluster'],
    palette='tab10',
    s=100
)
plt.title('Customer Segments using DBSCAN')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='DBSCAN Cluster')
plt.grid(True)
plt.show()
avg_spending_kmeans = df.groupby('Cluster')['Spending Score (1-100)'].mean()
print("\nAverage Spending per KMeans Cluster:")
print(avg_spending_kmeans)
avg_spending_dbscan = df.groupby('DBSCAN_Cluster')['Spending Score (1-100)'].mean()
print("\nAverage Spending per DBSCAN Cluster:")
print(avg_spending_dbscan)

df.to_csv('segmented_customers.csv', index=False)
df.to_excel('segmented_customers.xlsx', index=False)
print("Customer Segmentation Completed and Results Saved ")
