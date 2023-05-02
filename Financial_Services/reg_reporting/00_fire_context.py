# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC <img src=https://databricks-web-files.s3.us-east-2.amazonaws.com/notebooks/fs-lakehouse-logo.png width="600px">
# MAGIC 
# MAGIC [![DBR](https://img.shields.io/badge/DBR-9.0-green)](https://github.com/databrickslabs/fsi-solution-accelerators/tree/main/reg_reporting)
# MAGIC [![COMPLEXITY](https://img.shields.io/badge/COMPLEXITY-201-orange)](https://github.com/databrickslabs/fsi-solution-accelerators/tree/main/reg_reporting)
# MAGIC [![POC](https://img.shields.io/badge/POC-3d-red)](https://github.com/databrickslabs/fsi-solution-accelerators/tree/main/reg_reporting)
# MAGIC 
# MAGIC 
# MAGIC *In today’s interconnected world, managing risk and regulatory compliance is an increasingly complex and costly endeavour.
# MAGIC Regulatory change has increased 500% since the 2008 global financial crisis and boosted regulatory costs in the process. 
# MAGIC Given the fines associated with non-compliance and SLA breaches (banks hit an all-time high in fines of $10 billion in 2019 for AML), 
# MAGIC processing reports has to proceed even if data is incomplete. On the other hand, a track record of poor data quality is also "fined" because of "insufficient controls". 
# MAGIC As a consequence, many FSIs are often left battling between poor data quality and strict SLA, balancing between data reliability and data timeliness. 
# MAGIC In this solution accelerator, we demonstrate how [Delta Live Tables](https://databricks.com/product/delta-live-tables) 
# MAGIC can guarantee the acquisition and processing of regulatory data in real time to accommodate regulatory SLAs and, 
# MAGIC coupled with [Delta Sharing](https://databricks.com/blog/2021/05/26/introducing-delta-sharing-an-open-protocol-for-secure-data-sharing.html), 
# MAGIC to provide analysts with a real time confidence in regulatory data being transmitted. 
# MAGIC Through these series notebooks and underlying code, we will demonstrate the benefits of a standardized data model for regulatory 
# MAGIC data coupled the flexibility of Delta Lake to guarantee both **reliability** and **timeliness** in the transmission, 
# MAGIC acquisition and calculation of data between regulatory systems in finance*
# MAGIC 
# MAGIC 
# MAGIC ___
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | FIRE                                   | Regulatory models       | Apache v2  | https://github.com/SuadeLabs/fire                   |
