# --------------------------Task 5 - Decision Trees and Random Forests---------------------

# 1. Setup, Load Data, and Preprocessing
# We will prepare the data for modeling by splitting it into training and testing sets. Since tree-based models (Decision Trees and Random Forests) are not sensitive to feature scaling, we will skip the scaling step.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# 1.1 Load the dataset
df = pd.read_csv('/Users/divi/workspace/Elevate Labs/AI_ML Intern/Task5/heart.csv')

print("--- Initial Data Structure ---")
df.info()
print("\nTarget Distribution (0: No Disease, 1: Disease):")
print(df['target'].value_counts())

# 1.2. Define X (features) and y (target)
X = df.drop('target', axis=1)
y = df['target']

# 1.3. Split data into train-test sets
# We use stratify=y to ensure the class distribution is maintained in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 2. Train a Decision Tree Classifier and Analyze Overfitting
# A Decision Tree model is trained and its performance is evaluated. An initial, unconstrained tree often overfits the training data.

# 2.1. Train an unconstrained Decision Tree Classifier
dt_unconstrained = DecisionTreeClassifier(random_state=42)
dt_unconstrained.fit(X_train, y_train)

# Predictions
y_train_pred_dt_u = dt_unconstrained.predict(X_train)
y_test_pred_dt_u = dt_unconstrained.predict(X_test)

# Evaluate
train_acc_u = accuracy_score(y_train, y_train_pred_dt_u)
test_acc_u = accuracy_score(y_test, y_test_pred_dt_u)

print("\n--- Unconstrained Decision Tree Evaluation ---")
print(f"Training Accuracy: {train_acc_u:.4f} (Likely overfit)")
print(f"Test Accuracy: {test_acc_u:.4f}")

# 2.2. --- Control Tree Depth to mitigate overfitting 
# Train a constrained Decision Tree (max_depth=5)
dt_constrained = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_constrained.fit(X_train, y_train)

# Evaluate constrained tree
y_train_pred_dt_c = dt_constrained.predict(X_train)
y_test_pred_dt_c = dt_constrained.predict(X_test)

train_acc_c = accuracy_score(y_train, y_train_pred_dt_c)
test_acc_c = accuracy_score(y_test, y_test_pred_dt_c)

print(f"\n--- Constrained Decision Tree (max_depth=5) Evaluation ---")
print(f"Training Accuracy: {train_acc_c:.4f}")
print(f"Test Accuracy: {test_acc_c:.4f}")

# 3.Visualize the Decision Tree
# Visualize the constrained Decision Tree to understand the decision rules and the splitting process.

# 3.1. Visualize the constrained Decision Tree (Hint: Visualize the tree)
plt.figure(figsize=(20, 10))
plot_tree(dt_constrained,
          feature_names=X.columns,
          class_names=['No Disease', 'Disease'],
          filled=True,
          rounded=True,
          impurity=False,
          fontsize=10)
plt.title("Decision Tree Classifier (max_depth=5)")
plt.show()

# 4. Train a Random Forest Classifier and Compare Accuracy
# A Random Forest (an ensemble method using Bagging) is trained to improve generalization and accuracy by averaging the predictions of multiple trees 

# 4.1. Train a Random Forest Classifier (Hint: Train a Random Forest and compare accuracy)
# We use 100 trees (n_estimators) with a controlled maximum depth.
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_test_pred_rf = rf_model.predict(X_test)

# Evaluate
rf_accuracy = accuracy_score(y_test, y_test_pred_rf)
print("\n--- Random Forest Classifier Evaluation ---")
print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")
print(f"Decision Tree (max_depth=5) Test Accuracy: {test_acc_c:.4f}")
print("\nComparison: Random Forest usually provides better generalization due to ensemble learning.")

# Detailed Classification Report
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_test_pred_rf, target_names=['No Disease', 'Disease']))

# 5. Interpret Feature Importances
# Random Forest intrinsically provides a measure of feature importance, indicating which features contributed most significantly to the classification decisions across all trees.
# 5.1. Extract and visualize feature importances

feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

print("\n--- Random Forest Feature Importances ---")
print(feature_importances_sorted)

# 5.2. Visualize Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_sorted.values, y=feature_importances_sorted.index, palette='viridis')
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# 6. Evaluate using Cross-Validation
# Cross-validation (CV) provides a more robust estimate of model performance than a single train/test split by averaging results over multiple folds.

# 6.1. Perform cross-validation on the Random Forest model
# Use 10-fold cross-validation (cv=10) on the entire dataset
cv_scores = cross_val_score(rf_model, X, y, cv=10, scoring='accuracy')

print("\n--- Cross-Validation Evaluation (Random Forest, 10 Folds) ---")
print(f"CV Scores for each fold: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation of CV Accuracy: {cv_scores.std():.4f}")
print("\nInterpretation: The mean CV accuracy gives a more reliable estimate of how the model performs on unseen data.")

# --------------------------End of Task 5--------------------------------