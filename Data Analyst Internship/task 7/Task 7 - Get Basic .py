
# --------~~~~~~-----Task 7 - Get Basic Sales Summary from a Tiny SQLite Database using Python
# 1. Setting Up the SQLite Database and Creating the Sales Table

#  Connect and Create the Database File
import sqlite3

# Connect to the database (creates it if it doesn't exist)
conn = sqlite3.connect("sales_data.db")
cursor = conn.cursor()


# --- FIX: Drop the table to clear old, faulty schema ---
# This ensures that if a table with an incorrect schema already exists, it is removed.
cursor.execute("DROP TABLE IF EXISTS sales;")
print("Existing 'sales' table (if any) dropped.")
# --------------------------------------------------------


# 2. Define and Create the 'sales' Table (Corrected Schema)
create_table_query = """
CREATE TABLE sales (
    sale_id INTEGER PRIMARY KEY,
    product TEXT NOT NULL,
    quantity INTEGER NOT NULL,  -- All columns are defined in lowercase
    price REAL NOT NULL,
    sale_date TEXT
);
"""

# Execute the table creation command
cursor.execute(create_table_query)
conn.commit() # Commit the transaction to save the table structure
print("Sales table created successfully.")

# 3. Inserting Sample Data into the Table
# Sample sales data
sales_data = [
    ('Laptop', 2, 1200.00, '2023-11-01'),
    ('Mouse', 5, 25.00, '2023-11-01'),
    ('Laptop', 1, 1250.00, '2023-11-02'),
    ('Keyboard', 3, 75.00, '2023-11-02'),
    ('Mouse', 10, 24.50, '2023-11-03'),
    ('Keyboard', 1, 75.00, '2023-11-03'),
]

# SQL command to insert data
# FIX IS HERE: Changed 'Sales' to 'sales' (lowercase)
insert_query = "INSERT INTO sales (product, quantity, price, sale_date) VALUES (?, ?, ?, ?)"

# 4. Execute the insertions
cursor.executemany(insert_query, sales_data)
conn.commit() # Commit the transaction to save the data
print(f"{len(sales_data)} records inserted into the sales table.")

# Closing the Connection
# Close the connection
conn.close()
print("Database connection closed. You can now run the summary query.")


# --------~~~~~~-----Task 7 - Get Basic Sales Summary from a Tiny SQLite Database using Pandas
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load SQLite Database and Connect ---
conn = sqlite3.connect("sales_data.db")
print("Successfully connected to sales_data.db.")

# --- 2. Run Basic SQL Query ---
# This query calculates the total quantity sold (total_qty) and 
# the total revenue (revenue) for each distinct product.
query = """
SELECT 
    product, 
    SUM(quantity) AS total_qty, 
    SUM(quantity * price) AS revenue 
FROM 
    sales 
GROUP BY 
    product;
"""
print("SQL Query defined.")

# --- 3. Load into Pandas DataFrame ---
# Execute the SQL query and load the results directly into a Pandas DataFrame.
df = pd.read_sql_query(query, conn)

# Close the database connection once data is fetched
conn.close()
print("Database connection closed.")


# --- 4. Print Results ---
print("\n--- Sales Summary Results ---")
print(df)


# --- 5. Plot Simple Bar Chart ---
# Use the 'product' column for the x-axis and 'revenue' for the y-axis.
plt.figure(figsize=(8, 5))
df.plot(kind='bar', x='product', y='revenue', legend=False)

# Add title and labels for clarity
plt.title('Total Revenue by Product')
plt.xlabel('Product')
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=0) # Keep product names horizontal
plt.tight_layout()

# --- 6. Save Chart  ---
# plt.savefig("sales_chart.png") 

# Display the chart
plt.show()