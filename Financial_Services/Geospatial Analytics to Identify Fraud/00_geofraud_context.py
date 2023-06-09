# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/geoscan-fraud on the `web-sync` branch. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/identifying-fraud.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fs-lakehouse-logo.png width="600px">
# MAGIC 
# MAGIC [![DBU](https://img.shields.io/badge/DBU-XL-red)]()
# MAGIC [![COMPLEXITY](https://img.shields.io/badge/COMPLEXITY-301-red)]()
# MAGIC 
# MAGIC *A large scale fraud prevention system is usually a complex ecosystem made of various controls (all with critical SLAs), a mix of traditional rules and AI and a patchwork of technologies between proprietary on-premises systems and open source cloud technologies. In a previous [solution accelerator](https://databricks.com/blog/2021/01/19/combining-rules-based-and-ai-models-to-combat-financial-fraud.html), we addressed the problem of blending rules with AI in a common orchestration layer powered by MLFlow. In this series of notebooks centered around geospatial analytics, we demonstrate how Lakehouse enables organizations to better understand customers behaviours, no longer based on who they are, but how they bank, no longer using a one-size-fits-all rule but a truly personalized AI. After all, identifying abnormal patterns can only be made possible if one first understands what a normal behaviour is, and doing so for millions of customers becomes a challenge that requires both data and AI combined into one platform. As part of this solution, we are releasing a new open source geospatial library, [GEOSCAN](https://github.com/databrickslabs/geoscan), to detect geospatial behaviours at massive scale, track customers patterns over time and detect anomalous card transactions*
# MAGIC 
# MAGIC ---
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC In the first notebook, we introduce [GEOSCAN](https://github.com/databrickslabs/geoscan), a novel approach to geospatial clustering. We will aim at learning user transactional behaviour based on synthetic transactions data in NYC. In a second notebook, we leverage this information to detect transactions deviating from the norm and explore different ways to surface these anomalies from an analytics environment to an online data store.
# MAGIC 
# MAGIC <img src=https://brysmiwasb.blob.core.windows.net/demos/geoscan/geoscan_architecture.png alt="logical_flow" width="800">

# COMMAND ----------

# MAGIC %md
# MAGIC In this series of notebooks and companion library, we will be using [H3](https://eng.uber.com/h3/), an Hexagonal Hierarchical Spatial Index developed by Uber to analyze large spatial data sets. Partitioning areas of the Earth into identifiable grid cells as per image below, we will leverage this technique to detect transactions happening in close vicinity from one another.
# MAGIC 
# MAGIC <img src="https://1fykyq3mdn5r21tpna3wkdyi-wpengine.netdna-ssl.com/wp-content/uploads/2018/06/image12.png" width=300>
# MAGIC <br>
# MAGIC [source](https://eng.uber.com/h3/)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | com.uber:h3:3.6.3                      | Uber geospatial library | Apache2    | https://github.com/uber/h3                          |
# MAGIC | h3                                     | Uber geospatial library | Apache2    | https://github.com/uber/h3-py                       |
# MAGIC | org.scala-graph:graph-core_2.12:1.12.5 | Scala graph             | Apache2    | https://github.com/scala-graph/scala-graph          |
# MAGIC | com.databricks.labs:geoscan:0.1      | Geoscan algorithm       | Databricks | https://github.com/databrickslabs/geoscan           |
# MAGIC | folium                                 | Geospatial visualization| MIT        | https://github.com/python-visualization/folium      |
# MAGIC | pybloomfiltermmap3                     | Bloom filter            | MIT        | https://github.com/prashnts/pybloomfiltermmap3      |

# COMMAND ----------


