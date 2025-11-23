# ------Task 3 - Linear Regression-------

# 1. Setup, Load Data, and Preprocessing
# The initial steps involve importing necessary libraries, loading the data, and preprocessing the features to ensure they are suitable for the linear regression model.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1.1 Load the dataset
df = pd.read_csv('/Users/divi/workspace/Elevate Labs/AI_ML Intern/Task3/Housing.csv')

print("--- Initial Dataset Info ---")
df.info()

# --- Preprocessing Step 1: Handle Binary Categorical Variables ---
# Convert 'yes'/'no' columns to 1/0
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))

# --- Preprocessing Step 2: Handle Multi-level Categorical Variables ---
# Use One-Hot Encoding for 'furnishingstatus'
furnishing_status = pd.get_dummies(df['furnishingstatus'], drop_first=True)
df = pd.concat([df, furnishing_status], axis=1)
df = df.drop('furnishingstatus', axis=1)

print("\n--- Processed Data Head ---")
print(df.head())

# 2. Define Features (X) and Target (y) and ScalingThe target variable ($\mathbf{y}$) is price, and all other columns are features ($\mathbf{X}$). We use StandardScaler to normalize the numerical features.
# Define X and y
y = df['price']
X = df.drop('price', axis=1)

# List of numerical columns to scale (excluding the binary/dummy variables)
num_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Initialize and apply StandardScaler
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

print("\n--- Features (X) after Scaling ---")
print(X.head())

# 2.1 Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 3. Multiple Linear Regression Model Training and Evaluation
# A Multiple Linear Regression model is trained using all available features, and its performance is evaluated using standard regression metrics.

# 3.1. Fit a Linear Regression model using sklearn.linear_model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# 3.2. Evaluate model using MAE, MSE, R^2
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation (Multiple Linear Regression) ---")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Mean Squared Error (MSE): ${mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R^2) Score: {r2:.4f}")

# 4. Interpretation and Visualization
# The coefficients are interpreted to understand the impact of each feature on the house price. A plot of actual vs. predicted values is used to visually assess model fit.

# 4.1. Interpret coefficients
coefficients = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': lr_model.coef_})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)

print("\n--- Model Coefficients (Impact on Price) ---")
print(coefficients)

# 4.2. Visualize Actual vs. Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
# Plot the ideal line (where y_pred = y_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual Prices vs. Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

# 4.3. Visualize the residuals (errors)
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals (Actual - Predicted)')
plt.show()

print('''Interpretation of Coefficients
The coefficients in the table indicate the expected change in price for a one-unit change in the corresponding feature, assuming all other features remain constant.

Positive Coefficients (e.g., area, airconditioning, prefarea): An increase in these features (or their presence) is associated with an increase in the house price.

Negative Coefficients (e.g., unfurnished, semi-furnished): Compared to the reference category (furnished, since we used drop_first=True), houses with unfurnished or semi-furnished status tend to have a lower price.

Since the continuous features (area, bedrooms, etc.) were Standardized (scaled to have a mean of 0 and standard deviation of 1), the coefficients for these features show their relative importance. For example, a one-standard deviation increase in area has a very strong positive impact on price, as shown by its large positive coefficient.''')