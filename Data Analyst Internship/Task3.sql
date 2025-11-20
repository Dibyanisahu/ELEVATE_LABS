# Create schema Ecom
USE Ecom;
# View all records
SELECT * FROM Ecom.data
LIMIT 50;
# Check nulls in all columns
SELECT 
    SUM(InvoiceNo IS NULL) AS InvoiceNo_nulls,
    SUM(StockCode IS NULL) AS StockCode_nulls,
    SUM(Description IS NULL) AS Description_nulls,
    SUM(Quantity IS NULL) AS Quantity_nulls,
    SUM(InvoiceDate IS NULL) AS InvoiceDate_nulls,
    SUM(UnitPrice IS NULL) AS UnitPrice_nulls,
    SUM(CustomerID IS NULL) AS CustomerID_nulls,
    SUM(Country IS NULL) AS Country_nulls
FROM data;

# Check duplicates for all records
SELECT 
InvoiceNo,StockCode, Description, Quantity,InvoiceDate,UnitPrice, CustomerID, Country, COUNT(*) AS Duplicates
FROM data
GROUP BY InvoiceNo,StockCode, Description, Quantity,InvoiceDate,UnitPrice, CustomerID, Country
HAVING COUNT(*)>1;

# Find the total no. of duplicate entries
SELECT 
    SUM(dup_count - 1) AS total_duplicate_rows
FROM (
    SELECT COUNT(*) AS dup_count
    FROM data
    GROUP BY 
        InvoiceNo, StockCode, Description, Quantity,
        InvoiceDate, UnitPrice, CustomerID, Country
    HAVING COUNT(*) > 1
) AS t;

# Remove duplicate entries:
# Creating a clean deduplicated table
CREATE TABLE data_clean AS    # data_clean is the new table with unique entries
SELECT DISTINCT
    InvoiceNo, StockCode, Description, Quantity,
    InvoiceDate, UnitPrice, CustomerID, Country
FROM data;

# Comparing no.of records in both tables
SELECT COUNT(*) as total_records
FROM data;
SELECT COUNT(*) as total_records FROM data_clean;

#check datatypes of all columns
DESCRIBE data_clean;

# Convert InvoiceDate into datetime format
ALTER TABLE data_clean
ADD COLUMN InvoiceDate_dt DATETIME;
# Added PK row-no to check if this works to do the safe update
ALTER TABLE data_clean
ADD COLUMN row_no INT NOT NULL AUTO_INCREMENT PRIMARY KEY;  # this method didnt work, so doing temporary work below


# Temporarily disabled safe update mode and converted date into m/d/yyyy format 
SET SQL_SAFE_UPDATES = 0;


UPDATE data_clean
SET InvoiceDate_dt = STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i');

SET SQL_SAFE_UPDATES = 1;

# View invoicedate format
SELECT InvoiceDate
FROM data_clean
LIMIT 10;



