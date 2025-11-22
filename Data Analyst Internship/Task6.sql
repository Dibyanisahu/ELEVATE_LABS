# Sales Trend analysis using aggregation
USE Ecom;
# Objective: Calculate monthly revenue (Total_Revenue) and order volume (Order_Volume) from the 'cleaned_data' table to analyze time trends.

SELECT DATE_FORMAT(InvoiceDate_dt, '%Y-%m') AS Yer_Month,  #1.Extract Year and Month for Grouping
ROUND(SUM(Quantity * UnitPrice),2) AS Total_Revenue,   # 2.Calculate Total Monthly Revenue (Sales = Quantity * UnitPrice)
COUNT(DISTINCT InvoiceNo) AS Order_Volume     # 3.Calculate Total Monthly Order Volume (Count of distinct Invoice Numbers)
FROM
    data_clean
WHERE
    Quantity > 0 AND UnitPrice > 0   # Filter out cancellations/returns (Quantity <= 0) and zero-price items (UnitPrice = 0)
-- to ensure only valid sales transactions are included.
GROUP BY    # Group by the formatted Year-Month string
    Yer_Month
ORDER BY   # Order chronologically
    Yer_Month ASC;