import pandas as pd

# Load a CSV file into a DataFrame
data = pd.read_csv('/Users/divi/workspace/Elevate Labs/Data Analyst Intern/cleaned_data.csv')
print("Dataset Loaded Successfully.")
# Perform basic data exploration
df = pd.DataFrame(data)
# Display the first few rows of the dataset
print("--- Data Information ---")
df.info()
# Display the first 5 rows
print("\n--- First 5 Rows of the Dataset ---")
print(df.head())
# Display the shape of the dataset
print("\n--- Dataset Shape ---")
print(df.shape)
# Check for missing values
print("\n--- Missing Values in Each Column ---")
print(df.isnull().sum())
# Display descriptive statistics        
print("\n--- Descriptive Statistics ---")
print(df.describe(include='all').T)
# Display data types of each column
print("\n--- Data Types of Each Column ---")
print(df.dtypes)
# Display unique values in each column
print("\n--- Unique Values in Each Column ---")
for column in df.columns:
    unique_values = df[column].nunique()
    print(f"{column}: {unique_values} unique values")
# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\n--- Number of Duplicate Rows: {duplicates} ---")  

     
# Display value counts for categorical columns
print("\n--- Value Counts for Categorical Columns ---")
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    print(f"\nValue counts for column '{column}':")
    print(df[column].value_counts().head(10))  # Display top 10 value counts    

# Convert 'InvoiceDate' to datetime object
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y')

# Feature Engineering: Create 'Sales' column
df['Sales'] = df['Quantity'] * df['UnitPrice']

# Data Cleaning: Filter out cancelled orders (Quantity <= 0) and zero-price items (UnitPrice = 0)
# for meaningful sales analysis.
df_sales = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()

print(f"Original Records: {len(df)}")
print(f"Cleaned Sales Records: {len(df_sales)}")

# Check for CustomerID null values in the cleaned data
print(f"\nMissing CustomerIDs in cleaned data: {df_sales['CustomerID'].isnull().sum()}")

# Drop rows where CustomerID is null (as CustomerID is often key for segmentation/RFM)
df_sales.dropna(subset=['CustomerID'], inplace=True)
df_sales['CustomerID'] = df_sales['CustomerID'].astype(int) # Convert to integer
print(f"Final Records after dropping null CustomerID: {len(df_sales)}")

# **Observation:**
# * The `InvoiceDate` was converted to datetime.
# * A new feature, **`Sales`** (Quantity * UnitPrice), was created.
# * Negative quantities (returns) and zero unit prices were **filtered out** to focus on valid transactions, reducing the dataset size.
# * Missing `CustomerID` values were dropped for cleaner customer-level analysis.

# Save the cleaned and feature-engineered dataset to a new CSV file
df_sales.to_csv('/Users/divi/workspace/Elevate Labs/Data Analyst Intern/cleansales_data.csv', index=False)
print("\nCleaned sales data saved to 'cleansales_data.csv'.")    

# The cleaned sales data is now ready for further analysis and visualization.
#Univariate Analysis

import matplotlib.pyplot as plt
import seaborn as sns

# Plot histogram for 'Sales'
plt.figure(figsize=(10, 6))
sns.histplot(df_sales['Sales'], bins=50, kde=True)
plt.title('Distribution of Sales per Transaction')
plt.xlabel('Sales (pounds)')
plt.xlim(0, 50) # Zooming in on the bulk of data for better visualization
plt.show()
# fig.savefig('hist_sales.png')   

# **Observation (Sales Histogram):**
# * The distribution is highly **right-skewed**, with most transactions having very low sales values (below $\pounds10$).
# * There is a long tail of very high-value transactions, indicating a few large purchases or outliers.  

#Top 10 Countries by Total Sales
country_sales = df_sales.groupby('Country')['Sales'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=country_sales.index, y=country_sales.values, palette='viridis')
plt.title('Top 10 Countries by Total Sales')
plt.xlabel('Country')

#  Total Sales (£)
plt.ylabel('Total Sales (£)') 

plt.xticks(rotation=45, ha='right')
plt.show()

# **Observation (Top Countries):**
# * The **United Kingdom** dominates the sales, far outpacing the second-highest country. This suggests the company is primarily a domestic retailer or has its largest market in the UK.
# * Other key markets include the Netherlands, EIRE (Ireland), Germany, and France.

# UnitPrice Boxplot
# Boxplot for UnitPrice to check for outliers
plt.figure(figsize=(8, 6))
sns.boxplot(y=df_sales['UnitPrice'])
plt.title('Boxplot of Unit Price')

plt.ylabel('Unit Price (£)') 

plt.ylim(0, 10) # Limiting y-axis for better visibility of the main data
plt.show()


# **Observation (Unit Price Boxplot):**
# * The majority of unit prices are concentrated below 5 pounds.
# * The presence of many extreme values (outliers) above the main box-and-whiskers suggests there are a few very expensive, unique items.


# ## 4. Bivariate and Multivariate Analysis


# c. Identify relationships and trends: Monthly Sales Trend
df_sales['InvoiceMonth'] = df_sales['InvoiceDate'].dt.to_period('M')
monthly_sales = df_sales.groupby('InvoiceMonth')['Sales'].sum()
monthly_sales.index = monthly_sales.index.astype(str) # Convert PeriodIndex to string for plotting

plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Sales Trend Over Time')
plt.xlabel('Month')
plt.ylabel('Total Sales (pounds £)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
# fig.savefig('line_monthly_sales.png')


# **Observation (Monthly Sales Trend):**
# * Sales show a clear **upward trend** over the year, peaking sharply in **November** (likely due to holiday/Black Friday sales).
# * There is a significant drop in December, which is typical if the data cuts off early in the month.


# Use sns.heatmap() to check correlation among numerical variables
numerical_cols = ['Quantity', 'UnitPrice', 'Sales']
correlation_matrix = df_sales[numerical_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()
# fig.savefig('heatmap_correlation.png')


# **Observation (Correlation Heatmap):**
# * **Quantity and Sales** have a relatively strong positive correlation ($\sim0.82$), which is expected as buying more items increases total sales.
# * **UnitPrice and Sales** have a very low correlation ($\sim0.08$), suggesting that the unit price of an item has little linear relationship with the overall sales of that transaction. High-volume purchases (high Quantity) are the primary driver of high Sales figures.


# Scatterplot of Quantity vs UnitPrice
# Taking a sample due to large dataset size
df_sample = df_sales.sample(n=50000, random_state=42)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='UnitPrice', y='Quantity', data=df_sample)
plt.title('Quantity vs. Unit Price (Sampled)')
plt.xlabel('Unit Price (pounds £)')
plt.ylabel('Quantity')
plt.xlim(0, 50) # Limit for better visualization
plt.ylim(0, 500) # Limit for better visualization
plt.show()
# fig.savefig('scatter_quantity_unitprice.png')

# **Observation (Quantity vs. Unit Price Scatterplot):**
# * Most transactions are clustered at **low Quantity and low Unit Price**.
# * There are scattered points showing either high Quantity (bulk orders) or high Unit Price (expensive items), but few transactions have both extremely high Quantity and Unit Price, suggesting bulk orders tend to be for cheaper items.