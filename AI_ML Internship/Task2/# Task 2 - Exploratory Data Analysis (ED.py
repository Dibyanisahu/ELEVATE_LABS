# -------------Task 2 - Exploratory Data Analysis (EDA)----------- #
# 1. Setup and Load Data
# Load the dataset and display initial rows and structure.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")

# 1. Load the dataset
df_eda = pd.read_csv('/Users/divi/workspace/Elevate Labs/AI_ML Intern/Walmart.csv')
df_eda['Date'] = pd.to_datetime(df_eda['Date'], format='%d-%m-%Y')

# Initial inspection
print("--- First 5 rows of the dataset ---")
print(df_eda.head())

print("\n--- Dataset Info ---")
df_eda.info()

# 2. Generate Summary Statistics
# Summary statistics (mean, median, std, etc.) are generated for numerical features.
# Generate summary statistics (Hint: Generate summary statistics)
print("--- Summary Statistics for Numerical Features ---")
print(df_eda.describe())

print("\n--- Summary Statistics for Categorical Feature (Holiday_Flag) ---")
print(df_eda['Holiday_Flag'].value_counts())

# 3. Time Series Analysis: Patterns and Trends
# Aggregate the Weekly_Sales by Date across all stores to observe overall trends over time.
# Aggregate weekly sales across all stores
total_weekly_sales = df_eda.groupby('Date')['Weekly_Sales'].sum().reset_index()

# Plot Weekly Sales over Time (Hint: Identify patterns, trends, or anomalies)
plt.figure(figsize=(15, 6))
plt.plot(total_weekly_sales['Date'], total_weekly_sales['Weekly_Sales'], color='teal')
plt.title('Total Weekly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales (in millions)')
plt.show()

# Visualize the effect of holidays
holiday_sales = df_eda.groupby('Holiday_Flag')['Weekly_Sales'].mean()
print("\n--- Average Weekly Sales: Holiday vs Non-Holiday ---")
print(holiday_sales)

plt.figure(figsize=(6, 4))
sns.barplot(x=holiday_sales.index, y=holiday_sales.values, palette=['skyblue', 'salmon'])
plt.title('Average Sales: Non-Holiday (0) vs Holiday (1)')
plt.xlabel('Holiday Flag')
plt.ylabel('Average Weekly Sales')
plt.show()

# 4. Univariate Analysis: Distributions and Outliers
# Histograms show the distribution, and boxplots help visualize central tendency and outliers for numerical features.

numerical_cols_eda = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

# Create histograms for numeric features 
plt.figure(figsize=(15, 8))
for i, col in enumerate(numerical_cols_eda):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df_eda[col], kde=True, bins=30, color='darkblue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Create boxplots for numeric features 
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols_eda):
    plt.subplot(1, 5, i + 1)
    sns.boxplot(y=df_eda[col], color='purple')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# 5. Relationship Analysis: Correlation Matrix
# A correlation matrix is used to understand the linear relationships between variables, especially with the target variable Weekly_Sales.
# Calculate the correlation matrix 
correlation_matrix = df_eda[numerical_cols_eda].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Key Features')
plt.show()

# Inferences from the correlation matrix
print("\n--- Inferences from Correlation Matrix (with Weekly_Sales) ---")
print(correlation_matrix['Weekly_Sales'].sort_values(ascending=False))

# ----------END OF TASK 2 - EDA ---------- #.py ---------- #

