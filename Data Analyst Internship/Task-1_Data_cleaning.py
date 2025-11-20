import pandas as pd

# Example: Load a CSV file into a DataFrame
df_dataset = pd.read_csv('/Users/divi/workspace/Elevate Labs/SQL Developer/data.csv',encoding='latin1')
print(df_dataset.head())
# Check for null values in each column
Nulls = df_dataset.isnull().sum()
print("Null values in each column:\n", Nulls)
#Check for duplicates 
Duplicates = df_dataset.duplicated().sum()
print("Number of duplicate rows:", Duplicates)
Cleaned_Dataset = df_dataset.drop_duplicates().reset_index(drop=True)
print("Dataset after removing duplicates:\n", Cleaned_Dataset.head())
# check duplicates after cleaning
print(Cleaned_Dataset.duplicated().sum())
# standardize date columns to dd-mm-yyyy format
Cleaned_Dataset['InvoiceDate'] = pd.to_datetime(Cleaned_Dataset['InvoiceDate'])
# Converting date to dd-mm-yyyy format
Cleaned_Dataset['InvoiceDate'] = Cleaned_Dataset['InvoiceDate'].dt.strftime('%d-%m-%Y')
#Checking data types of each column
print(Cleaned_Dataset.dtypes)

# convert invoice number to integer
Cleaned_Dataset['InvoiceNo'] = pd.to_numeric(Cleaned_Dataset['InvoiceNo'], errors='coerce').fillna(0).astype(int)
# Convert stock coede to string
Cleaned_Dataset['StockCode'] = Cleaned_Dataset['StockCode'].astype(str)
#convert customer id to integer
Cleaned_Dataset['CustomerID'] = pd.to_numeric(Cleaned_Dataset['CustomerID'], errors='coerce').fillna(0).astype(int)
print(Cleaned_Dataset.dtypes)
# Save cleaned dataset to a new CSV file    
Cleaned_Dataset.to_csv('/Users/divi/workspace/Elevate Labs/Data Analyst Intern/cleaned_data.csv', index=False)
  


