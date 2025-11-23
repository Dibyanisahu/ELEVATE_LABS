# -------------------------------Task 6 - K-Nearest Neighbors (KNN) Classification--------------------------------------

# 1. Setup, Load Data, and Preprocessing (Normalization)
# KNN is a distance-based algorithm, which makes normalization/standardization crucial to prevent features with larger scales (e.g., Sepal Length) from dominating the distance calculation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1.1 Load the dataset
df = pd.read_csv('/Users/divi/workspace/Elevate Labs/AI_ML Intern/Task6/Iris.csv')

# Drop the 'Id' column as it is just an index
df = df.drop('Id', axis=1)

print("--- Initial Data Structure ---")
df.info()
print("\nTarget Variable (Species) Distribution:")
print(df['Species'].value_counts())

# 1.2. Define X (features) and y (target)
X = df.drop('Species', axis=1)
y = df['Species']

# Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalize/Standardize features using StandardScaler
# Use StandardScaler for Z-score normalization (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set shape: {X_train_scaled.shape}")
print(f"Testing set shape: {X_test_scaled.shape}")

# 2. Experiment with Different Values of K.
# The optimal value of K (the number of neighbors) is critical for KNN performance. We iterate through a range of $K$ values to find the one that yields the highest accuracy.

# 2.1. Experiment with different values of K (from 1 to 26) and record the accuracy for each K.
k_values = range(1, 26)
accuracies = []

for k in k_values:
    # Initialize KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Train the model
    knn.fit(X_train_scaled, y_train)
    # Predict on the test set
    y_pred = knn.predict(X_test_scaled)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# 2.2. Visualize the relationship between K and Accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='--', color='blue')
plt.title('KNN Accuracy vs. K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Test Accuracy')
plt.xticks(k_values)
plt.grid(True)

# 2.3. Find the optimal K value
optimal_k = k_values[np.argmax(accuracies)]
max_accuracy = np.max(accuracies)
plt.axvline(x=optimal_k, color='red', linestyle='-', label=f'Optimal K = {optimal_k}')
plt.legend()
plt.show()

print(f"\nOptimal K value found: {optimal_k}")
print(f"Maximum Test Accuracy at Optimal K: {max_accuracy:.4f}")

# 3. Train and Evaluate the Optimal KNN Model. 
# We train the final KNN model using the optimal K value identified in the previous step and evaluate its performance using accuracy and the confusion matrix.

# 3.1. Use KNeighbors Classifier from sklearn (with optimal K)
#  Evaluate model using accuracy, confusion matrix 
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train_scaled, y_train)

# Make predictions
y_pred_optimal = knn_optimal.predict(X_test_scaled)

# 3.2. Calculate Accuracy
final_accuracy = accuracy_score(y_test, y_pred_optimal)

# 3.3. Calculate Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred_optimal)

print("\n--- Optimal KNN Model Evaluation (K={}) ---".format(optimal_k))
print(f"Final Test Accuracy: {final_accuracy:.4f}")

# 3.4. Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_optimal))

# 3.5. Visualize the Confusion Matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Greens',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title('Confusion Matrix for Optimal KNN (K={})'.format(optimal_k))
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# 4. Visualize Decision Boundaries (2 Features)
# To visualize the decision boundaries, we must simplify the problem by selecting only two features (e.g., Petal Length and Petal Width). This allows us to plot the classification regions in 2D space.

# 4.1. Visualize decision boundaries using only 2 features
# Select two features for 2D visualization: Petal Length and Petal Width
X_2d = df[['PetalLengthCm', 'PetalWidthCm']]
y_2d = df['Species']

# Split and Scale the 2D data
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y_2d, test_size=0.3, random_state=42, stratify=y_2d)
scaler_2d = StandardScaler()
X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
X_test_2d_scaled = scaler_2d.transform(X_test_2d)

# 4.2. Train a KNN model on the 2D scaled data (using optimal_k)
knn_2d = KNeighborsClassifier(n_neighbors=optimal_k)
knn_2d.fit(X_train_2d_scaled, y_train_2d)

# 4.3. Create a mesh grid to plot boundaries
h = .02  # step size in the mesh
x_min, x_max = X_train_2d_scaled[:, 0].min() - 1, X_train_2d_scaled[:, 0].max() + 1
y_min, y_max = X_train_2d_scaled[:, 1].min() - 1, X_train_2d_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 4.4. Predict class for each point in the mesh
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 4.5. Map species names to integers for plotting
species_to_int = {name: i for i, name in enumerate(np.unique(y_2d))}
int_to_species = {i: name for name, i in species_to_int.items()}
Z_int = np.array([species_to_int[z] for z in Z.ravel()]).reshape(xx.shape)

# 4.6. Plot the decision boundaries
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z_int, alpha=0.3, cmap=plt.cm.RdYlBu)

# 4.7. Plot the training points (using scaled coordinates)
scatter = plt.scatter(X_train_2d_scaled[:, 0], X_train_2d_scaled[:, 1],
                      c=y_train_2d.map(species_to_int), edgecolors='k', cmap=plt.cm.RdYlBu)

plt.title('KNN Decision Boundaries (Petal Length vs Petal Width, K={})'.format(optimal_k))
plt.xlabel('Petal Length (Standardized)')
plt.ylabel('Petal Width (Standardized)')
plt.legend(handles=scatter.legend_elements()[0], labels=int_to_species.values(), title="Species")
plt.show()


# ----------------------------------End of Task 6------------------------------------------------------