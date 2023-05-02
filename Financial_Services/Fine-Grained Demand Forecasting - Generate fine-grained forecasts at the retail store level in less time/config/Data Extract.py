# Databricks notebook source
# MAGIC %md The purpose of this notebook is to download and set up the data we will use for the solution accelerator. Before running this notebook, make sure you have entered your own credentials for Kaggle.

# COMMAND ----------

# MAGIC %pip install kaggle

# COMMAND ----------

# MAGIC %md 
# MAGIC Set Kaggle credential configuration values in the block below: You can set up a [secret scope](https://docs.databricks.com/security/secrets/secret-scopes.html) to manage credentials used in notebooks. For the block below, we have manually set up the `solution-accelerator-cicd` secret scope and saved our credentials there for internal testing purposes.

# COMMAND ----------

# orignal

# import os
# # os.environ['kaggle_username'] = 'YOUR KAGGLE USERNAME HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
# os.environ['kaggle_username'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_username")

# # os.environ['kaggle_key'] = 'YOUR KAGGLE KEY HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
# os.environ['kaggle_key'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_key")

# COMMAND ----------

# shashank

# go to kaggle -> "your profile" from the right -> go to "Account" tab -> under the "API" section -> click in "Create New Token"

# kaggle_username -> shank234234
# kaggle_key -> 7fd2047453e181b5d38bd28c4ad0184f

import os

os.environ['kaggle_username'] = "shank234234"
os.environ['kaggle_key'] = "7fd2047453e181b5d38bd28c4ad0184f"

# you can add the credentials from the databricks secrets scope

# shashank

# COMMAND ----------



# COMMAND ----------

# MAGIC %md Download the data from Kaggle using the credentials set above:

# COMMAND ----------

# shashank
# you also need to install the kaggle library
!pip install kaggle
# shashank

# COMMAND ----------

# You need to accept the competition rules 

# 1) Go to the competition page on Kaggle.
# https://www.kaggle.com/competitions/demand-forecasting-kernels-only/rules
# 2) Scroll down to the "Rules" section.
# 3) Read the rules carefully.
# 4) If you agree to the rules, check the box next to "I understand and accept the rules."
# 5) Click the "I Accept" button to accept the rules.

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p ~/.kaggle/
# MAGIC 
# MAGIC echo $kaggle_username
# MAGIC echo $kaggle_key
# MAGIC 
# MAGIC json_data='{"username":"'"$kaggle_username"'","key":"'"$kaggle_key"'"}'
# MAGIC 
# MAGIC chmod 600 ~/.kaggle/kaggle.json
# MAGIC 
# MAGIC cd /databricks/driver
# MAGIC 
# MAGIC kaggle competitions download -c demand-forecasting-kernels-only --force --path /databricks/driver
# MAGIC 
# MAGIC unzip -o demand-forecasting-kernels-only.zip

# COMMAND ----------

# MAGIC %md Move the downloaded data to the folder used throughout the accelerator:

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/train.csv", "dbfs:/tmp/solacc/demand_forecast/train/train.csv")

# COMMAND ----------

# DBTITLE 1,Set up user-scoped database location to avoid conflicts
import re
from pathlib import Path
# Creating user-specific paths and database names
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
username_sql_compatible = re.sub('\W', '_', useremail.split('@')[0])
tmp_data_path = f"/tmp/fine_grain_forecast/data/{useremail}/"
database_name = f"fine_grain_forecast_{username_sql_compatible}"

# Create user-scoped environment
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name} LOCATION '{tmp_data_path}'")
spark.sql(f"USE {database_name}")
Path(tmp_data_path).mkdir(parents=True, exist_ok=True)
