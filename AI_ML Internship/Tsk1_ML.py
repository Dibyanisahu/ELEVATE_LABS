# Setup and Initial Data Exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Load a CSV file into a DataFrame
dataset = pd.read_csv('/Users/divi/workspace/Elevate Labs/AI_ML Intern/Walmart.csv',encoding='latin1')

# 2. Explore basic info (nulls, data types)
print("--- Initial Dataset Info ---")
dataset.info()

print("\n--- Missing Values Check ---")
print(dataset.isnull().sum())

print("\n--- Descriptive Statistics ---")
print(dataset.describe())

# Observation:
#There are no missing values in the dataset (.isnull().sum() shows all zeros).
#The Date column is of type object and needs to be converted to a datetime object for analysis.

# 2. Feature Engineering and Data Type Conversion
# The Date column is converted to datetime, and new temporal features (Year, Month, Day) are extracted to perform model building.

# Convert 'Date' column to datetime objects
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%d-%m-%Y')

# Extract temporal features (Feature Engineering)
dataset['Year'] = dataset['Date'].dt.year
dataset['Month'] = dataset['Date'].dt.month
dataset['Day'] = dataset['Date'].dt.day
dataset.info()

# The 'Date' column is no longer needed after extracting features
df = dataset.drop('Date', axis=1)

print("--- DataFrame after Feature Engineering ---")
df.info()
print(df.head())

#3. Handling Categorical Variables (Encoding)
# The Holiday_Flag is a binary (0 or 1) variable and is already in a numerical format. No additional encoding is strictly necessary, but if we wanted to one-hot encode the Store column (which is a common practice for nominal categories), the code would be:

# 'Holiday_Flag' (0/1) is already numerical.
# If we treat 'Store' as a nominal categorical variable and One-Hot Encode it. Then code as follows:

# Store_Encoded = pd.get_dummies(df['Store'], prefix='Store')
# df = pd.concat([df.drop('Store', axis=1), Store_Encoded], axis=1)

# For this task, we will keep 'Store' as is since it represents different entities,but the concept is demonstrated here.
print("\n--- Categorical Variable Status ---")
print("Holiday_Flag (0/1) is already numerical.")
print("Store is kept as is for this analysis.")

# 4. Outlier Visualization and Removal
# Outliers are visualized using boxplots and removed from the Weekly_Sales feature using the Interquartile Range (IQR) method.
# Identify numerical columns for outlier analysis
numerical_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

# Visualize outliers using boxplots 
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 5, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Remove outliers from 'Weekly_Sales' using the IQR method 
Q1 = df['Weekly_Sales'].quantile(0.25)
Q3 = df['Weekly_Sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Make an explicit copy after filtering to avoid SettingWithCopyWarning
df_cleaned = df.loc[(df['Weekly_Sales'] >= lower_bound) & (df['Weekly_Sales'] <= upper_bound)].copy()

print(f"\nOriginal shape: {df.shape}")
print(f"Cleaned shape after outlier removal: {df_cleaned.shape}")

# Verify outlier removal with a new boxplot for Weekly_Sales
plt.figure(figsize=(5, 5))
sns.boxplot(y=df_cleaned['Weekly_Sales'])
plt.title('Weekly_Sales after Outlier Removal')
plt.show()

# 5. Normalization/Standardization
# The remaining continuous numerical features are scaled using StandardScaler (Z-score normalization).


# Identify columns to scale (excluding Store and the binary/ordinal features)
cols_to_scale = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the selected columns (use .loc to be explicit about column assignment)
df_cleaned.loc[:, cols_to_scale] = scaler.fit_transform(df_cleaned[cols_to_scale])

print("\n--- Final Preprocessed Data Sample (Scaled) ---")
print(df_cleaned.head())
print(df_cleaned.describe().T)

# The dataset is now preprocessed and ready for machine learning model building.


