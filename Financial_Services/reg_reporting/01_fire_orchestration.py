# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline orchestration
# MAGIC Although defined here as a notebook, Delta Live Tables are designed to be operated as a job (this notebook will not work interactively and must be scheduled as a job through the Delta Live Tables interface as reported in the screenshot below). We will make use of a custom pyspark [library](https://github.com/aamend/fire) to interpret regulatory data model (FIRE) into delta lake compatible operations. This library will be brought here as a `%pip install` dependency. 
# MAGIC 
# MAGIC [![DLT](https://img.shields.io/badge/-DLT-grey)]()

# COMMAND ----------

# MAGIC %pip install git+https://github.com/aamend/fire.git

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://databricks-web-files.s3.us-east-2.amazonaws.com/notebooks/reg_reporting/images/pipeline_configuration.png" width=300>

# COMMAND ----------

# MAGIC %md
# MAGIC We retrieve the name of the entity to get the FIRE data model for as well as the directory (distributed file storage) where we expect new raw files to land. These parameters are passed to the delta live table notebook via job configuration as per the screenshot above.

# COMMAND ----------

# from pyspark import SparkConf

# # Create a SparkConf object
# conf = SparkConf()

# # Set a configuration parameter
# conf.set("spark.my.param", "my_value")

# # Get a configuration parameter
# my_param = conf.get("spark.my.param")
# print(my_param)

# COMMAND ----------

config = {
  "fire_entity"         : "collateral",
  "pipeline_dir"        : "/FileStore/antoine.amend@databricks.com/fire/collateral",
  "data_sharing_path"   : "/FileStore/antoine.amend@databricks.com/fire/share/collateral",
  "data_sharing_table"  : "fire_pipeline.collateral",
  "metrics_table"       : "fire_pipeline.ods"
}

# COMMAND ----------

# shashank

from pyspark import SparkConf

# Create a SparkConf object
conf = SparkConf()

conf.set("fire_entity", "collateral")

conf.set("landing_zone","/FileStore/legend/raw/collateral")
conf.set("invalid_format_path","/FileStore/legend/invalid/collateral")

conf.set("file_format", "json")
conf.set("max_files",3)

# COMMAND ----------

try:
  # the name of the fire entity we want to process
  fire_entity = conf.get("fire_entity")
except:
  raise Exception("Please provide [fire_entity] as job configuration")

try:
  # where new data file will be received
  landing_zone = conf.get("landing_zone")
except:
  raise Exception("Please provide [landing_zone] as job configuration")
  
try:
  # where corrupted data file will be stored
  invalid_format_path = conf.get("invalid_format_path")
except:
  raise Exception("Please provide [invalid_format_path] as job configuration")

try:
  # format we ingest raw data
  file_format = conf.get("file_format", "json")
except:
  raise Exception("Please provide [file_format] as job configuration")

try:
  # number of new file to read at each iteration
  max_files = int(conf.get("max_files", "1"))
except:
  raise Exception("Please provide [max_files] as job configuration")

# COMMAND ----------

# shashank

!pip install dlt

# COMMAND ----------

# !rm -rf /local_disk0/.ephemeral_nfs/envs/pythonEnv-d8cadcf8-6a6f-4012-b448-c7278d1947e0/bin/dlt
# !rm -rf /local_disk0/.ephemeral_nfs/envs/pythonEnv-d8cadcf8-6a6f-4012-b448-c7278d1947e0/lib/python3.9/site-packages/LICENSE.txt
# !rm -rf /local_disk0/.ephemeral_nfs/envs/pythonEnv-d8cadcf8-6a6f-4012-b448-c7278d1947e0/lib/python3.9/site-packages/README.md
# !rm -rf /local_disk0/.ephemeral_nfs/envs/pythonEnv-d8cadcf8-6a6f-4012-b448-c7278d1947e0/lib/python3.9/site-packages/dlt-0.2.5.dist-info/*
# !rm -rf /local_disk0/.ephemeral_nfs/envs/pythonEnv-d8cadcf8-6a6f-4012-b448-c7278d1947e0/lib/python3.9/site-packages/dlt/*

# COMMAND ----------

# !pip show dlt
# !pip uninstall dlt
# !pip install dlt==1.0.0
# !pip install dlt>=0.2.0
# !pip uninstall dlt
# !pip install dlt>=0.2.0
!pip list
# !/local_disk0/.ephemeral_nfs/envs/pythonEnv-d8cadcf8-6a6f-4012-b448-c7278d1947e0/bin/python -m pip install --upgrade pip


# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import udf
import dlt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schematizing
# MAGIC Even though records may sometimes "look" structured (e.g. JSON files), enforcing a schema is not just a good practice; in enterprise settings, and especially relevant in the space of regulatory compliance, it guarantees any missing field is still expected, unexpected fields are discarded and data types are fully evaluated (e.g. a date should be treated as a date object and not a string). Using FIRE pyspark module, we retrieve the spark schema required to process a given FIRE entity (e.g. collateral) that we apply on a stream of raw records. This process is called data schematization.

# COMMAND ----------

from fire.spark import FireModel
fire_model = FireModel().load(fire_entity)
fire_schema = fire_model.schema

# COMMAND ----------

# MAGIC %md
# MAGIC Our first step is to retrieve files landing to a distributed file storage using Spark auto-loader (though this framework can easily be extended to read different streams, using a Kafka connector for instance). In continuous mode, files will be processed as they land, `max_files` at a time. In triggered mode, only new files will be processed since last run. Using Delta Live Tables, we ensure the execution and processing of delta increments, preventing organizations from having to maintain complex checkpointing mechanisms to understand what data needs to be processed next; delta live tables seamlessly handles records that haven't yet been processed, first in first out.

# COMMAND ----------

@dlt.create_table()
def bronze():
  return (
    spark
      .readStream
      .option("maxFilesPerTrigger", str(max_files))
      .option("badRecordsPath", invalid_format_path)
      .format(file_format)
      .schema(fire_model.schema)
      .load(landing_zone)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Expectations
# MAGIC Applying a schema is one thing, enforcing its constraints is another. Given the schema definition of a FIRE entity, we can detect if a field is required or not. Given an enumeration object, we ensure its values consistency (e.g. country code). In addition to the technical constraints derived from the schema itself, the FIRE model also reports business expectations using e.g. `minimum`, `maximum`, `maxItems` JSON parameters. All these technical and business constraints will be programmatically retrieved from the FIRE model and interpreted as a series of SQL expressions. 

# COMMAND ----------

from fire.spark import FireModel
fire_model = FireModel().load(fire_entity)
fire_constraints = fire_model.constraints

# COMMAND ----------

# MAGIC %md
# MAGIC Our pipeline will evaluate our series of SQL rules against our schematized dataset (i.e. reading from Bronze), dropping record breaching any of our expectations through the `expect_all_or_drop` pattern and reporting on data quality in real time (note that one could simply flag records or fail an entire pipeline using resp. `expect_all` or `expect_all_or_fail`). At any point in time, we have clear visibility in how many records were dropped prior to landing on our silver layer.

# COMMAND ----------

@dlt.create_table()
@dlt.expect_all_or_drop(dict(zip(fire_constraints, fire_constraints)))
def silver():
  return dlt.read_stream("bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Invalid records
# MAGIC In this example, we made the choice to explicitly isolate invalid from valid records to ensure 100% data quality of regulatory data being transmitted. But in order to ensure full compliance (quality AND volume), we should also redirect invalid records to a quarantine table that can be further investigated and replayed if needed.

# COMMAND ----------

@udf("array<string>")
def failed_expectations(expectations):
  # retrieve the name of each failed expectation 
  return [name for name, success in zip(fire_constraints, expectations) if not success]

# COMMAND ----------

# MAGIC %md
# MAGIC Using a simple user defined function, we add an additional field to our original table with the name(s) of failed SQL expressions. The filtered output is sent to a quarantine table so that the union of quarantine and silver equals the volume expected from our bronze layer.

# COMMAND ----------

@dlt.create_table()
def quarantine():
  return (
      dlt
        .read_stream("bronze")
        .withColumn("_fire", F.array([F.expr(value) for value in fire_constraints]))
        .withColumn("_fire", failed_expectations("_fire"))
        .filter(F.size("_fire") > 0)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Take away
# MAGIC Finally, our pipeline has been orchestrated between Bronze, Silver and Quarantine, ensuring reliability in the transmission and validation of regulatory reports as new records unfold. As represented in the screenshot below, risk analysts have full visibility around number of records being processed in real time. In this specific example, we ensured that our collateral entity is exactly 92.2% complete (quarantine handles the remaining 8%).
# MAGIC 
# MAGIC <img src="https://databricks-web-files.s3.us-east-2.amazonaws.com/notebooks/reg_reporting/images/pipeline_processing.png">

# COMMAND ----------

# MAGIC %md
# MAGIC In the next section, we will demonstrate how organizations can create a simple operation data store to consume delta live tables metrics in real time, as new regulatory data is transmitted. Finally, we will demonstrate how delta sharing capability can ensure integrity in the reports beeing exchanged between FSIs and regulatory bodies.
