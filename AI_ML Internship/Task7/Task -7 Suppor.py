# ------------------------Task -7 Support Vector Machines-------------------------
# In this task, we will implement a Support Vector Machine (SVM) classifier using Python in Breast Cancer Dataset.

# 1. Setup and Data Loading 
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# Load the dataset
# The dataset file is 'breast-cancer.csv'
df = pd.read_csv('/Users/divi/workspace/Elevate Labs/AI_ML Intern/Task7/breast-cancer.csv')

# Display the first few rows and check for null values
print("--- Dataset Head ---")
print(df.head())
print("\n--- Dataset Info ---")
df.info()

# The 'id' column is irrelevant for modeling and 'Unnamed: 32' seems to be a redundant or empty column 
# Check for a column that is entirely NaN or unnecessary like 'Unnamed: 32' if it exists.
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
if 'Unnamed: 32' in df.columns:
    df.drop('Unnamed: 32', axis=1, inplace=True)

# Encode the target variable 'diagnosis' (M=Malignant, B=Benign) to numerical values (M=1, B=0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

print(f"\nTarget variable distribution:\n{df['diagnosis'].value_counts()}")


# 2. Data Preprocessing (Separation and Scaling)
# The features (X) and target (y) are separated, and all feature data is scaled to ensure fair weighting for the SVM algorithm.

# Separate features (X) and target (y)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split the data into training and testing sets (using cross-validation is mentioned, but a basic split is needed first)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. Training SVM with Linear and RBF Kernels
# The task requires training SVM models with both linear and RBF kernels.

# --- Linear Kernel SVM ---
# Train an SVM model with a linear kernel (simulating linear classification)
linear_svc = SVC(kernel='linear', random_state=42)
linear_svc.fit(X_train_scaled, y_train)

# Predict and evaluate the linear SVM
y_pred_linear = linear_svc.predict(X_test_scaled)
print("--- Linear Kernel SVM Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_linear):.4f}")
print(classification_report(y_test, y_pred_linear))

# --- RBF (Non-linear) Kernel SVM ---
# Train an SVM model with a radial basis function (RBF) kernel (for non-linear classification)
rbf_svc = SVC(kernel='rbf', random_state=42)
rbf_svc.fit(X_train_scaled, y_train)

# Predict and evaluate the RBF SVM
y_pred_rbf = rbf_svc.predict(X_test_scaled)
print("--- RBF Kernel SVM (Default) Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rbf):.4f}")
print(classification_report(y_test, y_pred_rbf))

# 4. Hyperparameter Tuning using Cross-Validation
# To optimize performance and handle overfitting, hyperparameters like C (regularization) and gamma (RBF kernel coefficient) are tuned using GridSearchCV and cross-validation.

# Define the parameter grid for RBF kernel
# C controls the trade-off between maximizing the margin and minimizing classification errors
# gamma defines how far the influence of a single training example reaches
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Use GridSearchCV with cross-validation (cv=5)
# This addresses the instruction to 'Use cross-validation to evaluate performance'
grid_search = GridSearchCV(SVC(random_state=42), param_grid, refit=True, verbose=2, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Display the best parameters and score
print("\n--- Hyperparameter Tuning Results (RBF Kernel) ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Evaluate the best model on the test set
best_svc = grid_search.best_estimator_
y_pred_tuned = best_svc.predict(X_test_scaled)

print("\n--- Tuned RBF Kernel SVM Test Set Results ---")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")
print(classification_report(y_test, y_pred_tuned))

# 5. Visualization of Decision Boundary (using PCA)
# For 2D visualization of the decision boundary, the high-dimensional data must be reduced. Principal Component Analysis (PCA) is used to project the data onto the first two principal components.

# Apply PCA to reduce data to 2 dimensions for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_train_scaled)

# Re-train the best SVM model on the 2D projected training data
X_2d_train, X_2d_test, y_2d_train, y_2d_test = train_test_split(X_2d, y_train, test_size=0.3, random_state=42, stratify=y_train)
svc_2d = SVC(kernel=best_svc.kernel, C=best_svc.C, gamma=best_svc.gamma, random_state=42)
svc_2d.fit(X_2d_train, y_2d_train)

# Function to plot the decision boundary
def plot_decision_boundary(X_2d_data, y_data, model, title):
    h = .02  # step size in the mesh
    x_min, x_max = X_2d_data[:, 0].min() - 1, X_2d_data[:, 0].max() + 1
    y_min, y_max = X_2d_data[:, 1].min() - 1, X_2d_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the class for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and the training points
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot the training points
    scatter = plt.scatter(X_2d_data[:, 0], X_2d_data[:, 1], c=y_data, cmap=plt.cm.coolwarm, edgecolors='k')
    
    # Highlight the Support Vectors
    if hasattr(model, 'support_vectors_'):
        plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                    s=200, facecolors='none', edgecolors='k', marker='o', label='Support Vectors')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.legend(handles=scatter.legend_elements()[0], labels=['Benign (0)', 'Malignant (1)'])
    plt.show()

# Visualize the decision boundary of the tuned RBF SVM on the PCA-reduced training data
plot_decision_boundary(X_2d_train, y_2d_train, svc_2d, 'SVM Decision Boundary (RBF Kernel) on PCA-Reduced Data ')

# ----------------------~~~~~---End of Task -7---~~~~~~----------------------