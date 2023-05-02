# Databricks notebook source
# MAGIC %md
# MAGIC %md 
# MAGIC # Reputation risk - ETL / anonymisation
# MAGIC **Nurturing Happier Customers with Data+AI**: *When it comes to the term "Risk Management", traditionally companies have seen guidance and frameworks around capital requirements from Basel standards. But, none of these guidelines mention Reputation Risk and for years organizations have lacked a clear eay to manage and measure Reputational Risk. Given how the conversation has shifted recently towards importance of ESG, companies must bridge the reputation-reality gap and ensure processes are in place to adapt to changing beliefs and expectations from stakeholders and customers.*
# MAGIC 
# MAGIC ---
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/01_rep_etl.html">STAGE1</a>: Using Delta Lake for ingesting anonymized customer complaints in real time
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/02_rep_eda.html">STAGE2</a>: Exploring complaints data at scale using Koalas and Pandas
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/03_rep_modelling.html">STAGE3</a>: Leverage AI to better operate customer complaints
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/04_rep_augmented.html">STAGE4</a>: Supercharge your BI reports with augmented intelligence
# MAGIC ---
# MAGIC <sri.ghattamaneni@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Context
# MAGIC 
# MAGIC In this notebook, we download records from [consumerfinance](http://files.consumerfinance.gov) and store CSV parsed data back to a delta table. We ensure invalid / malformed records are kept in quarantine for further inspection. We also address a key challenge in securing unstructured data where each complaint could be a transcript from audio call, web chat, e-mail and contain personal information such as customer first and last names. We demonstrate how organisations can leverage natural language processing (NLP) techniques to anonymize highly unstructured records whilst preserving their semantic value (i.e. replacing a mention of name should preserve the grammatical meaning of a consumer complaint). 
# MAGIC 
# MAGIC ### Dependencies
# MAGIC As reported in below cell, we use multiple 3rd party libraries that must be made available across Spark cluster. Assuming you are running this notebook on a Databricks cluster that does not make use of the ML runtime, you can use `dbutils.library.installPyPI()` utility to install python libraries in that specific notebook context. For java based libraries, or if you are using an 7.X ML runtime, please follow these [alternative steps](https://docs.databricks.com/libraries.html#workspace-library) to load libraries to your environment.

# COMMAND ----------

# shank
# dbutils.library.installPyPI command has been deprecated

!pip install spacy

# COMMAND ----------

# DBTITLE 1,Install libraries
# shank
# dbutils.library.installPyPI command has been deprecated

# dbutils.library.installPyPI("spacy")

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Create a database
# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS complaints

# COMMAND ----------

# MAGIC %md
# MAGIC As Databricks already leverages all the security tools provided by the cloud vendors, Apache SparkTM and Delta Lake offer additional enhancements such as Data Quarantine and Schema Enforcement to maintain and protect the quality of data in a timely manner. We will be using Spark to read in the complaints data by using a schema and persist it to Delta Lake. In this process we also provide a path to bad records which may be caused due to schema mismatch, data corruption or syntax errors into a separate location which could then be investigated later for consistency.

# COMMAND ----------

# DBTITLE 1,Download customer complaints
# MAGIC %sh 
# MAGIC wget http://files.consumerfinance.gov/ccdb/complaints.csv.zip -O /tmp/complaints.csv.zip
# MAGIC unzip -p /tmp/complaints.csv.zip > /dbfs/tmp/complaints.csv

# COMMAND ----------

# DBTITLE 1,Create quarantine folder
try:
  dbutils.fs.rm("/tmp/complaints_invalid", True)
except:
  pass

dbutils.fs.mkdirs("/tmp/complaints_invalid")

# COMMAND ----------

# DBTITLE 1,Define schema
from pyspark.sql.types import *
  
schema = StructType(
  [
    StructField("Date received", DateType(), True),
    StructField("Product", StringType(), True),
    StructField("Sub-product", StringType(), True),
    StructField("Issue", StringType(), True),
    StructField("Sub-issue", StringType(), True),
    StructField("Consumer complaint narrative", StringType(), True),
    StructField("Company public response", StringType(), True),
    StructField("Company", StringType(), True),
    StructField("State", StringType(), True),
    StructField("ZIP code", StringType(), True),
    StructField("Tags", StringType(), True),
    StructField("Consumer consent provided?", StringType(), True),
    StructField("Submitted via", StringType(), True),
    StructField("Date sent to company", DateType(), True),
    StructField("Company response to consumer", StringType(), True),
    StructField("Timely response?", StringType(), True),
    StructField("Consumer disputed?", StringType(), True),
    StructField("Complaint ID", LongType(), True)
  ]
)

# COMMAND ----------

# # 

# # valid_products = ["Consumer Loan", "Debt collection", "Mortgage","Credit card"]

# df = spark \
#   .read \
#   .option("header", "true") \
#   .option("delimiter", ",") \
#   .option("quote", "\"") \
#   .option("escape", "\"") \
#   .option("badRecordsPath", "/tmp/complaints_invalid") \
#   .schema(schema) \
#   .csv("/tmp/complaints.csv") \
#   .filter(F.col("Complaint ID").isNotNull()) \
#   .filter(F.length(F.col("Consumer complaint narrative")) > 20) \
# #   .filter(F.col("Product").isin(valid_products))


# COMMAND ----------

# from pyspark.sql import functions as F

# df = spark \
#   .read \
#   .option("header", "true") \
#   .option("delimiter", ",") \
#   .option("quote", "\"") \
#   .option("escape", "\"") \
#   .option("badRecordsPath", "/tmp/complaints_invalid") \
#   .schema(schema) \
#   .csv("/tmp/complaints.csv") \
#   .filter(F.col("Complaint ID").isNotNull()) \
#   .filter(F.length(F.col("Consumer complaint narrative")) > 20)


# display(df)

# COMMAND ----------

# display(df.select("Product").distinct())

# COMMAND ----------

# for i in df.select("Product").distinct().collect():
#     print(i)

# COMMAND ----------

# df.select("Product").distinct().show()

# COMMAND ----------

# DBTITLE 1,Store complaints on Delta Lake
from pyspark.sql import functions as F
from pyspark.sql.functions import lit
  
# read original dataframe and handle bad records
df = spark \
  .read \
  .option("header", "true") \
  .option("delimiter", ",") \
  .option("quote", "\"") \
  .option("escape", "\"") \
  .option("badRecordsPath", "/tmp/complaints_invalid") \
  .schema(schema) \
  .csv("/tmp/complaints.csv") \
  .filter(F.col("Complaint ID").isNotNull()) \
  .filter(F.length(F.col("Consumer complaint narrative")) > 20) \
#   .filter(F.col("Product").isin(classes))

# persist records to Delta table
df.select(
  F.col("Complaint ID").alias("complaint_id"),
  F.col("Company").alias("company"),
  F.col("Date received").alias("received_date"),
  F.col("Product").alias("product"),
  F.col("Issue").alias("issue"),
  F.col("Consumer complaint narrative").alias("complaint"),
  F.col("State").alias("state"),
  F.when(F.col("Timely response?") == "Yes", lit(1)).otherwise(lit(0)).alias("timely_response"),
  F.when(F.col("Consumer disputed?") == "Yes", lit(1)).otherwise(lit(0)).alias("consumer_disputed")
).write.mode("overwrite").format("delta").saveAsTable("complaints.complaints_bronze")

# COMMAND ----------

# DBTITLE 1,150K complaints
# MAGIC %sql
# MAGIC SELECT * FROM complaints.complaints_bronze

# COMMAND ----------

# MAGIC %md
# MAGIC ## Handling bad records
# MAGIC As part of the data ingestion process, we provided a path to bad records (`badRecordsPath`) which may be caused due to schema mismatch, data corruption or syntax errors into a separate location which could then be investigated later for consistency.

# COMMAND ----------

# DBTITLE 1,Bad records
display(dbutils.fs.ls("/tmp/complaints_invalid"))

# COMMAND ----------

# DBTITLE 1,Data quarantine
# Commented for the latest data
# display(spark.read.json("/tmp/complaints_invalid/20200930T132854/bad_records/"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Anonymising content
# MAGIC Ensuring no PII and complying with PCI-DSS regulation is common for financial services. Whilst it is relatively easy to remove, tokenise or anonymize sensitive information from structured data (e.g. tokenising card number using well known digit patterns), it becomes a real challenge with unstructured data. Large banking institutions can easily process a million of customer complaints a year, where each complaint could be a transcript from audio call, web chat, e-mail or scanned letter and contain personal information such as customer first and last names.
# MAGIC 
# MAGIC The first approach any bank uses when cleaning unstructured data it to search for well defined patterns, such as a 16 digits card number or a 9 digits social security number, replacing any potential PII or card number with e.g. `XXXXX-XXXXX-XXXX`. However, searching for names becomes *searching for unstructured pattern in an unstructured data*, and replacing names with `YYYY YYYYY` would break the text grammatical meaning (probably annoying most of data scientists along the way). We demonstrate here how banks can leverage databricks to detect names through NLP and anonymize highly unstructured records whilst preserving content semantic value (i.e. replacing a mention of name should preserve grammatical meaning). 

# COMMAND ----------

# DBTITLE 1,Download pre-trained models
import spacy
from spacy import displacy
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# COMMAND ----------

# DBTITLE 1,Before anonymization
original = 'My name is Antoine Amend. I have recently been called by Sri Ghattamaneni (a debt collector) in an attempt to retrieve payment for a debt. I have notified Junta Nakai, the creditor of the debt, that I have no responsibility to the account and that it was opened fraudulently but I have not had any success in resolving this matter thus far.'

doc = nlp(original)
htmlOriginal = displacy.render(doc, style='ent', jupyter=False)
displayHTML(htmlOriginal)

# COMMAND ----------

# DBTITLE 1,After anonymization
anonymized = original

for X in doc.ents:
  if(X.label_ == 'PERSON'):
    anonymized = anonymized.replace(X.text, "John Doe")

doc_a = nlp(anonymized)
htmlAnonymized = displacy.render(doc_a, style='ent', jupyter=False)
displayHTML(htmlAnonymized)

# COMMAND ----------

# MAGIC %md
# MAGIC Although this is just an example, one could make this business logic much more complex by generating multiple random names and desambiguating `Mr. Amend` from `Antoine Amend` so that not only the grammatical structure is respected, but the semantic value is fully preserved.

# COMMAND ----------

# DBTITLE 1,Create user defined functions
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
import pandas as pd
from typing import Iterator
from pyspark.sql import functions as F
import spacy
import re

def anonymize_record(original, nlp):
  """
  Simple text anonymisation
  Given a text, we extract entities from spacy library and replace mention of person
  One can make that logic as complex as they see fit and further enrich with regex pattern matching for e.g. credit card or social sdecurity number
  """
  doc = nlp(original)
  
  # One can add multiple regexes or any pattern for sensitive information e.g. PCI / PII
  # Alternatively, one may have to create their own NER tagger by manually annotating data 
  original = re.sub(r"(x{2,})[\-\/\s]?\1", "9999", original, flags=re.I)
  
  for X in doc.ents:
    if(X.label_ == 'PERSON'):
      original = original.replace(X.text, "John Doe")
  return original
    
@pandas_udf('string')
def anonymize(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:

  # load spacy model only once
  spacy.cli.download("en_core_web_sm")
  nlp = spacy.load("en_core_web_sm")
  
  # Extract organisations for a batch of content 
  for content_series in content_series_iter:
    yield content_series.map(lambda x: anonymize_record(x, nlp))
    
spark.udf.register("anonymize", anonymize)

# COMMAND ----------

# DBTITLE 1,Anonymize on demand
# MAGIC %sql
# MAGIC SELECT
# MAGIC   complaint,
# MAGIC   anonymize(complaint) AS complaint_anonymized
# MAGIC FROM
# MAGIC   complaints.complaints_bronze

# COMMAND ----------

# DBTITLE 1,Maintain anonymized version using auto loader
# Reading as a stream, processing record since last check point
# Reading from a delta table
input_stream = spark \
  .readStream \
  .format("delta") \
  .table("complaints.complaints_bronze")
  
# Anonymize text content
output_stream = input_stream.withColumn("complaint", anonymize(F.col("complaint")))
  
# Create a streaming job triggered only once that only processes data since last checkpoint
# Write anonymized records to delta table for ML / analytics
output_stream \
  .writeStream \
  .trigger(once=True) \
  .option("checkpointLocation", "/tmp/complaints_checkpoint") \
  .format("delta") \
  .table("complaints.complaints_bronze_anonymized")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/01_rep_etl.html">STAGE1</a>: Using Delta Lake for ingesting anonymized customer complaints in real time
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/02_rep_eda.html">STAGE2</a>: Exploring complaints data at scale using Koalas and Pandas
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/03_rep_modelling.html">STAGE3</a>: Leverage AI to better operate customer complaints
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/04_rep_augmented.html">STAGE4</a>: Supercharge your BI reports with augmented intelligence
# MAGIC ---
