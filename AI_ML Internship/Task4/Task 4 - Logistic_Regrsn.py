# -------Task 4 - Classification with Logistic Regression------
# 1. Setup, Load Data, and Preprocessing - Import libraries, load the dataset, and preprocess features by encoding the categorical target variable and scaling the numerical features.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 1.1. Load the dataset
df = pd.read_csv('/Users/divi/workspace/Elevate Labs/AI_ML Intern/Task4/data.csv')

# Drop the 'id' column as it is not a feature for prediction
# Drop the last column (unnamed) if it exists, as it is often empty in this dataset version
if df.columns[-1].startswith('Unnamed'):
    df = df.iloc[:, :-1]

df = df.drop('id', axis=1)

print("--- Initial Data Structure ---")
df.info()

# --- Preprocessing Step 1: Encode the Target Variable ---
# The target variable 'diagnosis' is categorical (M/B). Convert it to numerical (M=1, B=0).
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
print("\nTarget Variable Mapping: Malignant (M) = 1, Benign (B) = 0")

# Define X (features) and y (target)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# 1.2. Train/test split and standardize features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize and apply StandardScaler to standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set shape: {X_train_scaled.shape}")
print(f"Testing set shape: {X_test_scaled.shape}")

# 2. Model Training and Prediction
# The Logistic Regression model is trained on the scaled training data.

# 2.1. Fit a Logistic Regression model
# Note: LogisticRegression is a linear model for classification.
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Make predictions (class labels) and prediction probabilities
y_pred = lr_model.predict(X_test_scaled)
y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Model Training Complete ---")

# 3. Model Evaluation
# The model is evaluated using key classification metrics: Confusion Matrix, Accuracy, Precision, Recall, and ROC-AUC.

# 3.1. Evaluate with confusion matrix, precision, recall, ROC-AUC

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_mat = confusion_matrix(y_test, y_pred)

print("\n--- Model Evaluation (Default Threshold: 0.5) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Visualize the Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Benign (0)', 'Predicted Malignant (1)'],
            yticklabels=['Actual Benign (0)', 'Actual Malignant (1)'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# 4. ROC Curve and Threshold Tuning
# The Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) are plotted to visualize classifier performance across different thresholds.

# 4.1. Calculate ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 4.2. Tune threshold (Example: Finding a threshold that maximizes Recall)

# In medical contexts, we often want to maximize Recall (True Positive Rate) to ensure we don't miss Malignant cases (False Negatives are minimized).
optimal_threshold_index = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_threshold_index]

print("\n--- Threshold Tuning Example ---")
print(f"Optimal Threshold (maximizing TPR - FPR): {optimal_threshold:.4f}")

# 4.3. Apply the new threshold
y_pred_tuned = (y_pred_proba >= optimal_threshold).astype(int)
tuned_recall = recall_score(y_test, y_pred_tuned)
tuned_precision = precision_score(y_test, y_pred_tuned)

print(f"Recall with tuned threshold: {tuned_recall:.4f}")
print(f"Precision with tuned threshold: {tuned_precision:.4f}")

# 5. Explaining the Sigmoid Function
# Logistic Regression uses the Sigmoid function (also known as the logistic function) to map any real-valued number into a probability (a value between 0 and 1).
# If the resulting probability is above a set threshold (default is 0.5), the prediction is classified as 1 (Malignant); otherwise, it is classified as 0 (Benign).

# 5.1. Explain sigmoid function (via code for visualization)
def sigmoid(z):
    """Sigmoid function definition."""
    return 1 / (1 + np.exp(-z))

# Generate a range of Z values (linear scores)
z_values = np.linspace(-10, 10, 100)
# Calculate the corresponding probability (P) values
p_values = sigmoid(z_values)

plt.figure(figsize=(8, 5))
plt.plot(z_values, p_values, color='green', lw=3)
plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.axvline(0, color='gray', linestyle='--', label='Z = 0')
plt.title('Sigmoid (Logistic) Function')
plt.xlabel('Linear Score (z)')
plt.ylabel('Probability (P)')
plt.legend()
plt.show()

# -------End of Task 4------