<img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/fs-lakehouse-logo-transparent.png width="600px">

[![DBR](https://img.shields.io/badge/DBR-10.4ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/10.4ml.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)
[![POC](https://img.shields.io/badge/POC-8_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

*In a previous [solution accelerator](https://github.com/databricks-industry-solutions/merchant-classification), we 
demonstrated the need for a Lakehouse architecture to address one of the key challenges in retail banking, 
merchant classification. With the ability to classify card transactions data with clear brands information, 
retail banks can leverage this data asset further to unlock deeper customer insights. Moving from a traditional 
segmentation approach based on demographics, income and credit history towards behavioral clustering based on 
transactional patterns, millions of underbanked users with limited credit history could join a more inclusive banking 
ecosystem. Loosely inspired from the excellent work from [Capital One](https://arxiv.org/pdf/1907.07225.pdf) and in 
line with our previous experience in large UK based retail banking institutions, this solution focuses on learning 
hidden relationships between customers based on their card transaction pattern. How similar or dissimilar two customers 
are based on the shops they visit?* 

___
<antoine.amend@databricks.com>

___


<img src=https://raw.githubusercontent.com/databricks-industry-solutions/transaction-embedding/main/images/reference_architecture.png width="1000px">

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| PyYAML                                 | Reading Yaml files      | MIT        | https://github.com/yaml/pyyaml                      |

