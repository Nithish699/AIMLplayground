# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Scikit-learn imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load dataset (Amazon Customer Clustering Model)
amazon_df = pd.read_excel("Dataset_master.xlsx", sheet_name="Amazon.com Clusturing Model ")

# Drop Customer ID (not useful for clustering)
amazon_df = amazon_df.drop(columns=["Cus_ID"])

# Encode categorical column ('Sex': M=0, F=1)
encoder = LabelEncoder()
amazon_df["Sex"] = encoder.fit_transform(amazon_df["Sex"])

# Feature Scaling using MinMaxScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(amazon_df)

# Convert scaled data back into a DataFrame
amazon_scaled_df = pd.DataFrame(scaled_data, columns=amazon_df.columns)

# Find the optimal number of clusters using the Elbow Method
wcss = []  # Within-cluster sum of squares
K_range = range(1, 11)  # Testing K from 1 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(amazon_scaled_df)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker="o", linestyle="-", color="b")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal K")
plt.show()

# Selecting optimal K based on the elbow curve
optimal_k = 4  # Set based on elbow method

# Training the K-Means model
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
amazon_scaled_df["Cluster"] = kmeans.fit_predict(amazon_scaled_df)

# Scatter plot for Age vs Income with clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=amazon_scaled_df["Age"], 
    y=amazon_scaled_df["Income"], 
    hue=amazon_scaled_df["Cluster"], 
    palette="viridis"
)
plt.xlabel("Age (Scaled)")
plt.ylabel("Income (Scaled)")
plt.title("Clustering Visualization: Age vs Income")
plt.legend(title="Cluster")
plt.show()

# Splitting dataset into Train & Test sets
X = amazon_scaled_df.drop(columns=["Cluster"])  # Features
y = amazon_scaled_df["Cluster"]  # Target labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict clusters on test data
y_pred = kmeans.predict(X_test)

# Print some predictions
print("Actual Clusters:", y_test.values[:10])
print("Predicted Clusters:", y_pred[:10])

# Evaluate clustering quality
sil_score = silhouette_score(X_test, y_pred)
db_index = davies_bouldin_score(X_test, y_pred)
ch_score = calinski_harabasz_score(X_test, y_pred)

# Print evaluation metrics
print(f"Silhouette Score: {sil_score:.3f}")  # Higher is better
print(f"Davies-Bouldin Index: {db_index:.3f}")  # Lower is better
print(f"Calinski-Harabasz Score: {ch_score:.3f}")  # Higher is better

# Save trained K-Means model and scaler
joblib.dump(kmeans, "amazon_kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")
