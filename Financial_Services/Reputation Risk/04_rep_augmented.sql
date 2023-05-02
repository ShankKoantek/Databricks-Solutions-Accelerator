-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Reputation risk - Augmented Intelligence
-- MAGIC **Nurturing Happier Customers with Data+AI**: *When it comes to the term "Risk Management", traditionally companies have seen guidance and frameworks around capital requirements from Basel standards. But, none of these guidelines mention Reputation Risk and for years organizations have lacked a clear eay to manage and measure Reputational Risk. Given how the conversation has shifted recently towards importance of ESG, companies must bridge the reputation-reality gap and ensure processes are in place to adapt to changing beliefs and expectations from stakeholders and customers.*
-- MAGIC 
-- MAGIC ---
-- MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/01_rep_etl.html">STAGE1</a>: Using Delta Lake for ingesting anonymized customer complaints in real time
-- MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/02_rep_eda.html">STAGE2</a>: Exploring complaints data at scale using Koalas and Pandas
-- MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/03_rep_modelling.html">STAGE3</a>: Leverage AI to better operate customer complaints
-- MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/04_rep_augmented.html">STAGE4</a>: Supercharge your BI reports with augmented intelligence
-- MAGIC ---
-- MAGIC <sri.ghattamaneni@databricks.com>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Context
-- MAGIC Whilst we can now apply ML developped in previous notebooks to automatically classify and re-reroute new complaints in real time, as they unfold, the possibility to utilize UDF in SQL code gives Business Analysts the ability to directly interact with our models while querying data for visualization. This can enable us in producing further actionable insights using Databricks’ notebook visualization or Redash. Using simple SQL syntax we can easily look at complaints attributed to different products over a period of time in a given location. We report some examples of queries below, all supercharged with our models registered on mlflow.

-- COMMAND ----------

-- DBTITLE 1,Register user defined ML functions
-- MAGIC %python
-- MAGIC import mlflow
-- MAGIC import mlflow.pyfunc
-- MAGIC from pyspark.sql.types import *
-- MAGIC 
-- MAGIC spark.udf.register("classify", mlflow.pyfunc.spark_udf(spark, "models:/complaint_classification/staging", StringType()))
-- MAGIC spark.udf.register("severity", mlflow.pyfunc.spark_udf(spark, "models:/complaint_severity/production", DoubleType()))

-- COMMAND ----------

-- DBTITLE 1,AI augmented view
SELECT 
  c.received_date, 
  c.complaint, 
  classify(c.complaint) AS product,
  severity(c.complaint) AS dispute
FROM 
  complaints.complaints_bronze_anonymized c

-- COMMAND ----------

-- DBTITLE 1,Complaints per products
SELECT 
  classify(c.complaint) AS product, 
  count(1) AS `number of complaints`
FROM 
  complaints.complaints_bronze_anonymized c
GROUP BY 
  classify(c.complaint)
ORDER BY 
  2 DESC

-- COMMAND ----------

-- DBTITLE 1,Number of complaints by state
SELECT 
  c.state, 
  count(1) AS `number of complaints`
FROM 
  complaints.complaints_bronze_anonymized c
GROUP BY 
  c.state
ORDER BY 
  2 DESC

-- COMMAND ----------

-- DBTITLE 1,Where are our most annoyed consumers
WITH severities AS (
  SELECT 
    c.state, 
    CASE
        WHEN severity(c.complaint) > 0.8 THEN 1
        ELSE 0
    END AS dispute
  FROM complaints.complaints_bronze_anonymized c
)


SELECT 
  state, 
  100 * SUM(dispute) / COUNT(1) AS `dispute_rate`
FROM 
  severities
GROUP BY 
  state

-- COMMAND ----------

-- DBTITLE 1,Indiana consumers disputing about debt collection, mainly
SELECT 
  classify(c.complaint) AS product, 
  count(1) AS `number of complaints`
FROM 
  complaints.complaints_bronze_anonymized c
WHERE 
  c.state = 'IN'
  AND severity(c.complaint) > 0.6
GROUP BY 
  classify(c.complaint)
ORDER BY 1 ASC

-- COMMAND ----------

-- DBTITLE 1,Number of complaints per month in Indiana
SELECT 
  year(c.received_date) AS `year`, 
  month(c.received_date) AS `month`, 
  classify(c.complaint) AS product, 
  count(1) AS `number of complaints`
FROM 
  complaints.complaints_bronze_anonymized c
WHERE
  c.state = 'IN'
GROUP BY 
  year(c.received_date), 
  month(c.received_date), 
  classify(c.complaint)
ORDER BY 
  1, 2

-- COMMAND ----------

-- DBTITLE 1,Proportion of disputed complaints in Indiana
SELECT 
  year(c.received_date) AS `year`, 
  month(c.received_date) AS `month`, 
  CASE WHEN severity(c.complaint) > 0.6 THEN 'disputed' ELSE 'non disputed' END AS dispute,
  count(1) AS `number of complaints`
FROM 
  complaints.complaints_bronze_anonymized c
WHERE
  c.state = 'IN'
GROUP BY 
  year(c.received_date), 
  month(c.received_date),
  dispute
ORDER BY 
  1, 2

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ---
-- MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/01_rep_etl.html">STAGE1</a>: Using Delta Lake for ingesting anonymized customer complaints in real time
-- MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/02_rep_eda.html">STAGE2</a>: Exploring complaints data at scale using Koalas and Pandas
-- MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/03_rep_modelling.html">STAGE3</a>: Leverage AI to better operate customer complaints
-- MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/04_rep_augmented.html">STAGE4</a>: Supercharge your BI reports with augmented intelligence
-- MAGIC ---
