# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/survival. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/survivorship-and-churn.

# COMMAND ----------

# MAGIC %md In the previous notebook, we saw that there were significant differences in the curves when compared across acquisition channel, initial payment method, and initial payment plan days. In this notebook, we'll take a look at how these variables interact to determine the risk that a customer will drop-out during each of the three observed at-risk periods.
# MAGIC 
# MAGIC To do this, we'll make use of a [Cox Proportional Hazards model](https://en.wikipedia.org/wiki/Proportional_hazards_model) (again made available through the [lifelines](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html) library).

# COMMAND ----------

# MAGIC %md ##Step 1: Prepare the Environment
# MAGIC 
# MAGIC As before, we need to load some libraries for our analysis:

# COMMAND ----------

# DBTITLE 1,Install Needed Libraries
# MAGIC %pip install lifelines

# COMMAND ----------

# DBTITLE 1,Load Needed Libraries
import pandas as pd 
import numpy as np

from lifelines.fitters.coxph_fitter import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from mlflow import sklearn
import shutil

# COMMAND ----------

# MAGIC %md We also need to load some data.  Let's limit all features, *i.e.* payment method, payment plan days & registration channel, to those with more than 10,000 members associated with each:
# MAGIC 
# MAGIC **NOTE** In the last notebook, we explicitly dropped registration channels 10, 13 and 16.  The 10,000 member criteria has the same effect on registration channel values.

# COMMAND ----------

# DBTITLE 1,Assemble Data For Modeling
sql_statement='''
WITH transaction_attributes AS (
    SELECT
      a.msno,
      a.trans_at,
      FIRST(b.payment_method_id) as payment_method_id,
      FIRST(b.payment_plan_days) as payment_plan_days
    FROM (  -- base transaction dates
      SELECT 
        msno,
        transaction_date as trans_at,
        MAX(membership_expire_date) as expires_at
      FROM kkbox.transactions
      GROUP BY msno, transaction_date
      ) a
    INNER JOIN kkbox.transactions b
      ON  a.msno=b.msno AND
          a.trans_at=b.transaction_date AND 
          a.expires_at=b.membership_expire_date
    WHERE b.payment_plan_days > 0
    GROUP BY
      a.msno,
      a.trans_at
    )
  SELECT
    m.duration_days,
    m.churned,
    n.payment_method_id as init_payment_method_id,
    n.payment_plan_days as init_payment_plan_days,
    COALESCE(o.registered_via, -1) as registered_via
  FROM kkbox.subscriptions m
  INNER JOIN transaction_attributes n
    ON m.msno=n.msno AND m.starts_at=n.trans_at
  LEFT OUTER JOIN kkbox.members o
    ON m.msno=o.msno
  WHERE 
    n.payment_plan_days IN (
      SELECT payment_plan_days 
      FROM transaction_attributes 
      GROUP BY payment_plan_days
      HAVING COUNT(DISTINCT msno)>10000
      ) AND
    n.payment_method_id IN (
      SELECT payment_method_id 
      FROM transaction_attributes 
      GROUP BY payment_method_id 
      HAVING COUNT(DISTINCT msno)>10000
      ) AND
    o.registered_via IN (
      SELECT registered_via
      FROM kkbox.members
      GROUP BY registered_via
      HAVING COUNT(DISTINCT msno)>10000
      )'''

subscriptions_pd = spark.sql(sql_statement).toPandas()
subscriptions_pd.head()

# COMMAND ----------

# MAGIC %md ##Step 2: Feature Engineering
# MAGIC 
# MAGIC We now have our base dataset.  It contains three categorical features, *i.e.* *registered_via*, *init_payment_method_id* and *init_payment_plan_days*. As is typical of most algorithms, we will want to [one-hot encode](https://en.wikipedia.org/wiki/One-hot) *registered_via* and *init_payment_method* for use in our model.  However, *init_payment_plan_days* will require slightly different treatment.  More on that later.  For now, let's focus on one-hot encoding.
# MAGIC 
# MAGIC We'll do this using pandas's get_dummies() method, but notice that we are **NOT DROPPING** any features generated by the encoding at this time.  We will come back to this in the subsequent step:

# COMMAND ----------

# DBTITLE 1,Encode Categorical Features
# encode the subscription attributes
encoded_pd = pd.get_dummies(
    subscriptions_pd,
    columns=['init_payment_method_id', 'registered_via'], 
    prefix=['method', 'channel'],
    drop_first=False
    )

encoded_pd.head()

# COMMAND ----------

# MAGIC %md We now have a bunch of additional features encoding the unique values in our categorical fields. For each categorical field, we typically drop one of the encoded features.  We will do this here, but we want to do so in a manner where we control which of the features is dropped.
# MAGIC 
# MAGIC To understand why, consider that the Cox Proportional Hazards model defines a baseline  model that calculates the risk of an event - churn in this case - 
# MAGIC occurring over time. Each attribute included in the model alters this risk in a fixed (*proportional*) manner.  When we drop one of our one-hot columns, the value that column represents becomes represented in the baseline.  (In the literature, the values represented within the baseline function are sometime referred to as *reference* values.)  
# MAGIC 
# MAGIC With this in mind, we could drop an arbitrary combination of encoded features from our model to eliminate the multi-collinearity issue created by one-hot encoding. But if we drop encoded features that align with a *typical* member of our subscriber population, we can create a baseline that's intuitively understood by our business users. For example, if our most popular channel (*registered_via = 7*) and payment method (*init_payment_method_id = 41*) occur frequently in combination with each other, then a subscriber with these three values for these attributes might intuitively be understood as a representative subscriber of the KKBox service.  If we drop the one-hot encoded features corresponding to *registered_via = 7* and *init_payment_method_id = 41*, we have a baseline aligned with our representative subscriber and everything else is a deviation away from this recognized baseline:

# COMMAND ----------

# DBTITLE 1,Address Multi-Collinearity
# drop unnecessary (baseline) fields
survival_pd = encoded_pd.drop(['method_41', 'channel_7'], axis=1)

# review dataset
survival_pd.head(10)

# COMMAND ----------

# MAGIC %md ##Step 3: Model the Hazard
# MAGIC 
# MAGIC To train a model, we pass the CoxPHFitter our dataset, identifying which fields represent the subscription duration and the churn status.  A key assumption of this model is that there is a baseline hazard that changes over time and all the other factors considered simply scale this time-dependent function.  As was noted in the exploratory analysis, there is a sizeable drop-out that occurs at the first renewal day as indicated by the init_payment_plan_days value.  This would indicate that the initial payment plan days cannot be applied against a baseline hazard but instead each plan would require its own curve.  To generate this, we simply tell the fitter to stratify its work across this field:

# COMMAND ----------

# DBTITLE 1,Train the Model
cph = CoxPHFitter(alpha=0.05) # 95% confidence interval
cph.fit(survival_pd, 'duration_days', 'churned', strata='init_payment_plan_days')
cph.summary

# COMMAND ----------

# MAGIC %md The model summary provides quite a bit of useful information but we should start with the p-scores associated with each feature.  If any are greater than or equal to 0.05, the associated feature should be considered statistically insignificant  (assuming a 95% threshold). We don't have any such factors here but if we did, we might remove those fields, effectively rolling them into the baseline. 
# MAGIC 
# MAGIC We now have a model with statistically significant hazards defined.  These hazards, modeled for each remaining feature, modify the time-dependent baseline hazard found within the model. If we examine the *exp(coef)* field in the first table of the summary, we can see by what factor each hazard affects it. 
# MAGIC 
# MAGIC To state this another way, the values in *exp(coef)* are simple multipliers (*aka* hazard ratios), something we'll explore more deeply in the next notebook. While the values may vary slightly between runs, consider that channel 4 increases the hazard (risk of churn) by a factor of 1.2 (or +20%). If a subscriber registered via channel 4 using a payment method of 40 (1.3), the combined hazard in this window is 56% above the baseline as 1.2 \* 1.3 = 1.56.
# MAGIC 
# MAGIC So, where is this baseline function?  Accessing the baseline_cumulative_hazard_ property of the trained model, we can retrieve the estimated baseline hazard by subscription day.  Notice that each field represents one of the init_payment_plan_day values which was used to form strata:

# COMMAND ----------

# DBTITLE 1,Access the Baseline Hazard
cph.baseline_cumulative_hazard_.head(30)

# COMMAND ----------

# MAGIC %md ##Step 4: Validating the Models
# MAGIC 
# MAGIC The Cox PH model assumes that the hazard associated with a feature does not vary over time.  We used this to justify the splitting of our model based on strata. A formal test of this assumption is provided through the check_assumptions() method on the model object.  This particular test has not been done here because we've separated our dataset into at-risk windows narrow enough that we would can assume a reasonable risk of there not being a violation.  
# MAGIC 
# MAGIC Even if the assumption was violated, more and more statisticians are making the case that models with light violations of the assumption [may still be valid](https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html) if the hazard ratios are taken as average hazards over the period for which the models were trained. 
# MAGIC 
# MAGIC That said, it's just one method call, so why not just run the test? With larger dataset such as the one we are using, the check_assumptions() method is incredibly time-consuming. One trick that may help with reducing the time for this check is to randomly sample your data, retrain the model, and test the assumption off the newly trained version.  This may be a reasonable shortcut to overcome processing constraints but in testing with this dataset, we could only get a reasonable runtime with a 2% random sample of our 3.1 million subscriptions. 

# COMMAND ----------

# MAGIC %md ##Step 5: Persist the Datasets and Models
# MAGIC 
# MAGIC In the next notebook, we will examine how we might operationalize our model.  Instead of recreating all the work we've done here, let's save the model (and dataframes for some validation work) so that we may quickly pick them back up in the next notebook:

# COMMAND ----------

# DBTITLE 1,Save the Data Frames
dataset_path = '/dbfs/tmp/kkbox-survival/tmp/datasets/'

# drop any old copies that might exist
shutil.rmtree(dataset_path, ignore_errors=True)
dbutils.fs.mkdirs(dataset_path[5:])

# save datasets to temp storage
subscriptions_pd.to_pickle(dataset_path + 'subscriptions_pd')
survival_pd.to_pickle(dataset_path + 'survival_pd')

# COMMAND ----------

# DBTITLE 1,Save the Model
model_path = '/dbfs/tmp/kkbox-survival/tmp/models/'

# drop any old copies that might exist
shutil.rmtree(model_path, ignore_errors=True)

# save models to temp storage
sklearn.save_model( cph, model_path + 'cph')
