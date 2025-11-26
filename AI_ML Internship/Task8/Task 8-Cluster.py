# -----------~~~~~~~~-----Task 8: Clustering with K-Means on Mall Customer Dataset-----~~~~~~~~-----------

# 1. Setup and Data Loading
# The task requires loading and preparing the Mall Customer dataset.

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
# The dataset file is 'Mall_Customers.csv'
df = pd.read_csv('/Users/divi/workspace/Elevate Labs/AI_ML Intern/Task8/Mall_Customers.csv')

# Display the first few rows
print("--- Dataset Head ---")
print(df.head())
print("\n--- Dataset Info ---")
df.info()

# Select the two relevant features for clustering: 'Annual Income (k$)' and 'Spending Score (1-100)'
# These features are typically used for customer segmentation as they offer a good visualization plane.
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print("\n--- Features for Clustering (Income & Spending) ---")
print(X.head())

# 2. Feature Scaling
# The features are scaled to prevent one feature (e.g., Annual Income) from disproportionately influencing the distance calculations in K-Means.
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for better handling (optional)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\n--- Scaled Features Head ---")
print(X_scaled_df.head())

# 3. Finding the Optimal Number of Clusters (K)
# The Elbow Method is used to determine the optimal number of clusters (K) by looking for the "elbow" point in the plot of the Within-Cluster Sum of Squares (WCSS) against the number of clusters.

# Use the Elbow Method to find the optimal K
wcss = []
# Test K values from 1 to 10
max_k = 11
for i in range(1, max_k):
    # Initialize KMeans
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    # Append the WCSS (inertia) for the current K
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method results
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(range(1, max_k))
plt.grid(True)
plt.show()

# Based on the plot, the 'elbow' (the point of diminishing returns) is usually visible.
# Assume k=5 is chosen from the visual inspection of the WCSS plot.
optimal_k = 5
print(f"Optimal K selected based on Elbow Method: {optimal_k}")

# 4. Performing K-Means Clustering
# K-Means is fitted with the optimal number of clusters, and cluster labels are assigned to the original data.

# Initialize and fit K-Means with the optimal K
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add the cluster labels back to the original dataframe
df['Cluster'] = cluster_labels
X['Cluster'] = cluster_labels

# Display the size of each cluster
print("\n--- Cluster Sizes ---")
print(df['Cluster'].value_counts().sort_index())

# Display the cluster centers (in scaled feature space)
cluster_centers_scaled = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns[:-1])
print("\n--- Cluster Centers (Scaled) ---")
print(cluster_centers_scaled)

# 5. Evaluating Clustering Performance
# The Silhouette Score is calculated to evaluate the quality of the clustering.

# Calculate the Silhouette Score
# The score measures how similar an object is to its own cluster compared to other clusters.
# Range is [-1, 1]. Closer to 1 is better.
score = silhouette_score(X_scaled, cluster_labels)
print(f"\nSilhouette Score for K={optimal_k}: {score:.4f}")

# 6. Cluster Visualization
# The clusters are visualized using a scatter plot, color-coding each point according to its assigned cluster label.

# Visualize the clusters on the original (unscaled) features
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df,
                palette='viridis', style='Cluster', s=100)

# Plot the cluster centers (inverse transformed back to original scale for visual context)
# Note: For accurate interpretation, centers should be plotted relative to the scaled data,
# but plotting them on the original axes is common for segmentation visuals.
# We will use the means of the features per cluster as a proxy for the unscaled centers.
cluster_centers_unscaled = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
plt.scatter(cluster_centers_unscaled['Annual Income (k$)'],
            cluster_centers_unscaled['Spending Score (1-100)'],
            marker='X', s=250, color='red', label='Centroids', edgecolors='black')

plt.title(f'Customer Segments using K-Means (K={optimal_k}) ')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

print("\n--- Customer Segments Analysis (Mean of Unscaled Features) ---")
print(cluster_centers_unscaled)

#-----------~~~~~~~~-----End of Task 8-----~~~~~~~~-----------
