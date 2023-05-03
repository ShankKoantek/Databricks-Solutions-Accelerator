# Databricks notebook source
# MAGIC %md
# MAGIC # Automated Patient Risk Adjustment and Medicare HCC Coding from Clinical Notes
# MAGIC 
# MAGIC 
# MAGIC <img src=https://hls-eng-data-public.s3.amazonaws.com/img/medicare-risk-hcc.png width=60%>
# MAGIC 
# MAGIC This series of notebooks is also available at www.databricks.com/solutions/accelerators/medicare-risk-adjustment and https://github.com/databricks-industry-solutions/medicare-risk-adjustment

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Medicare Risk Adjustment: 
# MAGIC In the United States, the Centers for Medicare & Medicaid Services sets reimbursement for private Medicare plan sponsors based on the assessed risk of their beneficiaries. Information found in unstructured medical records may be more indicative of member risk than existing structured data, creating more accurate risk pools.
# MAGIC 
# MAGIC #### Why HCCs matter?
# MAGIC 
# MAGIC CMS-HCC risk scores factor heavily into benchmark calculations for MSSP ACOs and capitation rates for MA plans.
# MAGIC 
# MAGIC For MSSP ACOs, higher risk scores for a population translate into a higher benchmark for expenditures, while lower risk scores translate into a lower benchmark. Having an accurate benchmark is vital in achieving shared savings. A benchmark that inadequately reflects the underlying health status of a population will be too low and will lead to expenditures that are higher than expected.
# MAGIC 
# MAGIC For MA plans, higher risk scores translate into higher PMPM payments, and lower risk scores translate into lower PMPM payments. MA programs may suffer financial losses if their HCC scores underestimate the degree of illness within their beneficiary population.
# MAGIC 
# MAGIC HCCs are also increasingly relevant to exchange-based insurance products. Starting in 2014, CMS began using a modified version of its CMS-HCC model, the Department of Health & Human Services (HHS)-HCC model, to calculate risk scores for individuals purchasing insurance plans on the federal health exchange. The HHS-HCC model employs multiple models based on beneficiary age and type of insurance plan, and the risk score it calculates is used to estimate financial liability for insurers who offer products on the exchange.4
# MAGIC 
# MAGIC For a nascent MSSP ACO, getting your benchmark right has immense financial implications, because benchmarks generally aren't adjusted upward but may be adjusted downward. Each year, CMS reviews claims data and recalculates risk scores. For newly assigned beneficiaries, both the demographic and risk components of the risk score are used to adjust the benchmark in that year. Beyond that year of attribution, the beneficiary becomes a continuously assigned beneficiary. For continuously assigned beneficiaries, only the demographic component of the risk score is used to adjust the benchmark, with one exception. If a continuously assigned beneficiary appears healthier over time by virtue of fewer assigned HCCs from year to year, the risk component of the risk score for that beneficiary may be adjusted downward.
# MAGIC 
# MAGIC For example, consider a newly attributed 79-year-old male with uncomplicated diabetes (ICD-10 code E11.9, which maps to HCC 19, Diabetes without complication) in year one who develops a diabetic mono-neuropathy (ICD-10 code E0841, which maps to HCC 18, Diabetes with chronic complications) in year two after turning 80 years old. His year-two risk score will reflect an increase in predicted expenditure related to his new demographic category (male age 80 to 84) but will not reflect an increase in predicted expenditure related to his increase in disease severity (from HCC 19 to HCC 18). This lack of acknowledgement by CMS that chronic conditions can and frequently do worsen over time (despite appropriate medical management) is understandably frustrating to providers.
# MAGIC 
# MAGIC In addition, it is not enough to correctly code the patient's diagnosis. The plan of care must support the diagnosis for each condition listed, and the condition must be reestablished, along with the appropriate care plan, each year. In other words, previous coding resets to zero if not reestablished. It may seem that the coding for amputation would carry forward from year to year, but that is not the case.
# MAGIC 
# MAGIC The implications of this methodology are important. For continuously assigned beneficiaries, the specificity and completeness of coding and supporting documentation must remain at least as specific as it was during the pre-attribution period to ensure that the appropriate level of risk is attributed and the benchmark is not adjusted downward. Each chronic medical problem must be coded and documented annually to be incorporated by CMS into annual risk score recalculations. Given the potential for turnover in the population attributed to an MSSP ACO, a likely best practice would be to methodically document and code the health status of all Medicare beneficiaries regardless of whether they are currently attributed to your ACO.
# MAGIC 
# MAGIC Source: https://www.aafp.org/fpm/2016/0900/p24.html#fpm20160900p24-bt1

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## How to calculate the RAF Risk Score?
# MAGIC 
# MAGIC Source: https://www.aafp.org/fpm/2016/0900/p24.html
# MAGIC 
# MAGIC Source: https://www.cms.gov/Medicare/Health-Plans/MedicareAdvtgSpecRateStats/Risk-Adjustors
# MAGIC 
# MAGIC CMS calculates a risk score, or “risk adjustment factor” (RAF) score, for each individual beneficiary and provides this information to each ACO quarterly. Deriving these scores from HCCs is relatively straightforward. Each demographic adjustment and HCC carries a score/weight within the model. Adding the weights together produces a risk score for that beneficiary.
# MAGIC 
# MAGIC The model is normalized to a value of 1.0. Risk scores generally **range between 0.9 and 1.7**, and beneficiaries with risk scores **less than 1.0 are considered relatively healthy**. Each year CMS publishes a “denominator” that assists in converting risk scores to dollar amounts. For example, in 2014 this denominator was $9,050. Multiplying the risk score by this denominator produces an estimated annual expenditure for a beneficiary. Similarly, multiplying the weight of an HCC by the denominator produces a dollar figure that represents the marginal contribution of that HCC to the overall estimated health expenditure.
# MAGIC 
# MAGIC ## Example 1:
# MAGIC 
# MAGIC Suppose the capitated rate the plan receives from CMS is $500 per member per month. That represents the starting point and that rate can be adjusted based on the HCC scores. If the patient has diabetes with complications but that is not coded or not fully coded, the payment impact is pronounced:
# MAGIC 
# MAGIC |     Scenario 1     |     |                        |                    Scenario 2                    |                                 |                        |                         Scenario 3                         |                                         |                        |
# MAGIC |:------------------:|:---:|:----------------------:|:------------------------------------------------:|:-------------------------------:|:----------------------:|:----------------------------------------------------------:|:---------------------------------------:|:----------------------:|
# MAGIC |                    | HCC | Risk Adjustment Factor |                                                  |               HCC               | Risk Adjustment Factor |                                                            |                   HCC                   | Risk Adjustment Factor |
# MAGIC | 72-Year-Old Female |     |          0.346         |                72-Year-Old Female                |                                 |          0.346         |                     72-Year-Old Female                     |                                         |          0.346         |
# MAGIC | Diabetes not coded |     |           ***          | E11.9 Type 2 diabetes mellitus w/o complications | HCC19 Diabetes w/o complication |          0.124         | E11.41 Type 2 diabetes mellitus w/ diabetic mononeuropathy | HCC18 Diabetes w/ chronic complications |          0.625         |
# MAGIC |      Total RAF     |     |          0.374         |                                                  |                                 |          0.478         |                                                            |                                         |          0.692         |
# MAGIC |  Payment per month |     |          0.346         |                                                  |                                 |          0.470         |                                                            |                                         |          0.971         |
# MAGIC |  Payment per year  |     |         $173.00        |                                                  |                                 |         $235.00        |                                                            |                                         |         $485.50        |
# MAGIC 
# MAGIC ## Example 2:
# MAGIC 
# MAGIC | Scenario 1                                                  |                                         |                        | Scenario 2                                                  |                                         |                        | Scenario 3                                                                                 |                                                                       |                        |
# MAGIC |-------------------------------------------------------------|-----------------------------------------|------------------------|-------------------------------------------------------------|-----------------------------------------|------------------------|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|------------------------|
# MAGIC |                                                             | HCC                                     | Risk Adjustment Factor |                                                             | HCC                                     | Risk Adjustment Factor |                                                                                            | HCC                                                                   | Risk Adjustment Factor |
# MAGIC | 72-Year-Old Female                                          |                                         | 0.346                  | 72-Year-Old Female                                          |                                         | 0.346                  | 72-Year-Old Female                                                                         |                                                                       | 0.346                  |
# MAGIC | E11.41 Type 2 diabetes mellitus w/ diabetic mono neuropathy | HCC18 Diabetes w/ chronic complications | 0.625                  | E11.41 Type 2 diabetes mellitus w/ diabetic mono neuropathy | HCC18 Diabetes w/ chronic complications | 0.625                  | E11.41 Type 2 diabetes mellitus w/ diabetic mono neuropathy                                | HCC18 Diabetes w/ chronic complications                               | 0.625                  |
# MAGIC |                                                             |                                         |                        | K50.00 Crohn’s disease of small intestine w/o complications | HCC35 Inflammatory Bowel Disease        | 0.279                  | K50.00 Crohn’s disease of small intestine w/o complications                                | HCC35 Inflammatory Bowel Disease                                      | 0.279                  |
# MAGIC |                                                             |                                         |                        |                                                             |                                         |                        | M05.60 Rheumatoid arthritis of unspecified site w/ involvement of other organs and systems | HCC40 Rheumatoid Arthritis and Inflammatory Connective Tissue Disease | 0.423                  |
# MAGIC | Total RAF                                                   |                                         | 0.971                  |                                                             |                                         | 1.250                  |                                                                                            |                                                                       | 1.688                  |
# MAGIC | Payment per month                                           |                                         | $485.50                |                                                             |                                         | $625.00                |                                                                                            |                                                                       | $844.00                |
# MAGIC | Payment per year                                            |                                         | $5,826.00              |                                                             |                                         | $7,500.00              |                                                                                            |                                                                       | $10,128.00             |
# MAGIC 
# MAGIC Source: https://www.asahq.org/quality-and-practice-management/managing-your-practice/timely-topics-in-payment-and-practice-management/an-introduction-to-hierarchical-condition-categories-hcc

# COMMAND ----------

# MAGIC %md
# MAGIC ### Solution Outline
# MAGIC In this solution accelerator, we use John Snow Labs' pre-trained NLP models for healthcare to extract ICD10 codes and associated HCC, gender at birth and age from clinical notes. We then calculate RAF score based on extracted entities from these notes, using JSL for healthcare pre-defined functions.
# MAGIC 
# MAGIC [![](https://mermaid.ink/img/pako:eNp9kcFqwzAMhl9F-NRCC9s1h0HbJIWdxrpbUoYaK6khsYOtbGxN32LPtWeaM7sQGMwXWT-f9AvpIiojSSSisdif4SUtNfjnhlMQvr_gZI3-pKBvQiAt_3JOtW9kg74NYQchpjFm8E99Y9qo7mdU9C0svoM2TO4I6_XDqCp5fweWnJlMR9gWv8ox4hMDY-M7kJ1RuyJIxwmaoSM2NMPSwuexVVb0yIo0g2PkwUV5Gxxq1TLZBM5VBagloHNkWRkd6RHyxeLRKJ1AbPOq5HJ5287UI7-taJZk8yQPVr013o1G2BdPcaLnTQ6HythpVrESHdkOlfTHvEyVpeAzdVSKxH8l1Ti0XIpSXz069BKZMqnYWJHU2DpaCRzYHD50JRK2A92gVKE_URep6w9ImbCu)](https://mermaid-js.github.io/mermaid-live-editor/edit/#pako:eNp9kcFqwzAMhl9F-NRCC9s1h0HbJIWdxrpbUoYaK6khsYOtbGxN32LPtWeaM7sQGMwXWT-f9AvpIiojSSSisdif4SUtNfjnhlMQvr_gZI3-pKBvQiAt_3JOtW9kg74NYQchpjFm8E99Y9qo7mdU9C0svoM2TO4I6_XDqCp5fweWnJlMR9gWv8ox4hMDY-M7kJ1RuyJIxwmaoSM2NMPSwuexVVb0yIo0g2PkwUV5Gxxq1TLZBM5VBagloHNkWRkd6RHyxeLRKJ1AbPOq5HJ5287UI7-taJZk8yQPVr013o1G2BdPcaLnTQ6HythpVrESHdkOlfTHvEyVpeAzdVSKxH8l1Ti0XIpSXz069BKZMqnYWJHU2DpaCRzYHD50JRK2A92gVKE_URep6w9ImbCu)

# COMMAND ----------

# MAGIC %md
# MAGIC ## License
# MAGIC Copyright / License info of the notebook. Copyright [2022] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library License|Library License URL|Library Source URL|
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
# MAGIC |Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
# MAGIC |Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
# MAGIC |BeautifulSoup|MIT License|https://www.crummy.com/software/BeautifulSoup/#Download|https://www.crummy.com/software/BeautifulSoup/bs4/download/|
# MAGIC |Requests|Apache License 2.0|https://github.com/psf/requests/blob/main/LICENSE|https://github.com/psf/requests|
# MAGIC |Spark NLP Display|Apache License 2.0|https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/LICENSE|https://github.com/JohnSnowLabs/spark-nlp-display|
# MAGIC |Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
# MAGIC |Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC |Author|
# MAGIC |-|
# MAGIC |Databricks Inc.|
# MAGIC |John Snow Labs Inc.|

# COMMAND ----------

# MAGIC %md
# MAGIC ## Disclaimers
# MAGIC Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.
