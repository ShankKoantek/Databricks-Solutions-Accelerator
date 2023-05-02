# Databricks notebook source
# MAGIC %md
# MAGIC # Reputation risk - AI enrichment
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
# MAGIC In this notebook, we build a simple scikit-learn pipeline to classify complaints into four major categories of products we see in t-sne plot in previous notebook and predict the severity of complaints by training on previously disputed claims. Whilst Delta Lake provides reliability and performance in your data, MLFlow provides efficiency and transparency to your insights. Every ML experiment will be tracked and hyper parameters automatically logged in a common place, resulting in artifacts one can trust and act upon.
# MAGIC 
# MAGIC ### Dependencies
# MAGIC As reported in below cell, we use multiple 3rd party libraries that must be made available across Spark cluster. Assuming you are running this notebook on a Databricks cluster that does not make use of the ML runtime, you can use `dbutils.library.installPyPI()` utility to install python libraries in that specific notebook context. For java based libraries, or if you are using an 7.X ML runtime, please follow these [alternative steps](https://docs.databricks.com/libraries.html#workspace-library) to load libraries to your environment.
# MAGIC Need DBR 7.x with the following deps installed

# COMMAND ----------

# DBTITLE 1,Install libraries
!pip install mlflow

# dbutils.library.installPyPI("mlflow")
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Classifying customer complaints
# MAGIC Our first model will categorize products from consumer complaints narrative.

# COMMAND ----------

# DBTITLE 1,Retrieve complaints
from pyspark.sql import functions as F

jp_df = spark \
  .read \
  .table("complaints.complaints_bronze_anonymized") \
  .select("product", "complaint", "consumer_disputed") \
  .toPandas()

display(jp_df)

# COMMAND ----------

# DBTITLE 1,Balance dataset
from sklearn.utils import resample
import pandas as pd
import numpy as np

df_debt = jp_df[jp_df['product'] == 'Debt collection']
df_mort = jp_df[jp_df['product'] == 'Mortgage']
df_cred = jp_df[jp_df['product'] == 'Credit card']
df_loan = jp_df[jp_df['product'] == 'Consumer Loan']

dfs = [df_debt, df_mort, df_cred, df_loan]
majority = np.min([df.shape[0] for df in dfs])

def sample(df, n):
  return resample(
    df, 
    replace=True,                  # sample with replacement
    n_samples=n,                   # to match majority class
    random_state=123               # reproducible results
  )              

# Combine minority class with downspampled majority class
dfs_sampled = [sample(df, majority) for df in dfs]
df_sampled = pd.concat(dfs_sampled)
df_sampled['product'].value_counts().plot(kind='bar')

# COMMAND ----------

# DBTITLE 1,Train / Test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder().fit(df_sampled['product'])
y = encoder.transform(df_sampled['product'])
X = df_sampled.drop('product', axis=1)['complaint']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training set contains {} records".format(X_train.shape[0]))
print("Testing set contains {} records".format(X_test.shape[0]))

# COMMAND ----------

# DBTITLE 1,Train classifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import TransformerMixin, BaseEstimator, clone
import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name='complaint_classifier_sklearn'):

  # get mlflow run Id
  run_id = mlflow.active_run().info.run_id
  
  # Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer
  count_vect = CountVectorizer(ngram_range=(2,2), min_df=3)
  tf_transformer = TfidfTransformer(use_idf=True)
  clf = MultinomialNB()

  # define pipeline
  pipeline = Pipeline([
    ('vect', count_vect),
    ('tfidf', tf_transformer),
    ('clf', clf)
  ])

  # Train pipeline
  pipeline.fit(X_train, y_train)  
  y_pred = pipeline.predict(X_test)
  accuracy = accuracy_score(y_pred, y_test)
  
  # Log pipeline to mlflow
  mlflow.sklearn.log_model(pipeline, "pipeline")
  mlflow.log_metric("accuracy", accuracy)
  
  # Log classification report
  clsf_report = pd.DataFrame(classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)).transpose()
  clsf_report.to_csv('classification_report.csv', index=True)
  mlflow.log_artifact('classification_report.csv')

clsf_report

# COMMAND ----------

# DBTITLE 1,Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

# create confusion matrix
pred_df = pd.DataFrame(zip(pipeline.predict(X_test), y_test), columns=['predicted', 'actual'])
confusion_matrix = pd.crosstab(pred_df['actual'], pred_df['predicted'], rownames=['label'], colnames=['prediction'])

# plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt='d')
plt.savefig("confusion_matrix.png")

# store confusion matrix alongside model on mlflow
client = mlflow.tracking.MlflowClient()
client.log_artifact(run_id, "confusion_matrix.png")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Although we do have a fully functional model, it expects a numpy array of text as an input and returns a simple label as output. In real life scenario, one may interact with model from a dataframe (having a column containing text) and expect the class name (not the label) as an output. Therefore, we will repackage our sklearn pipeline as a `pyfunc` object to support this extra business logic.

# COMMAND ----------

# DBTITLE 1,Package pipeline as a pyfunc
class PyfuncClassifier(mlflow.pyfunc.PythonModel):
  
  def __init__(self, pipeline):
    self.pipeline = pipeline
    
  def load_context(self, context):
    # Load definition of labels
    with open(context.artifacts['label_path'], "r") as f:
      self.labels = f.read().split(",")
      
  def predict(self, context, df):
    # We expect a single column dataframe as an input
    X = df[df.columns[0]]
    y = [self.labels[i] for i in self.pipeline.predict(X)]
    return y
  
# We ensure that pyfunc has registered sklearn as dependency
conda_env = mlflow.pyfunc.get_default_conda_env()
conda_env['dependencies'][2]['pip'] += ['scikit-learn']

# We register our uber model as another mlflow experiment
with mlflow.start_run(run_name='complaint_classifier_pyfunc'):
  
  # persist labels
  label_path = "/tmp/labels.csv"
  with open(label_path, "w") as f:
    f.write(','.join(list(encoder.classes_)))
  
  # log model
  mlflow.pyfunc.log_model(
    'pipeline', 
    python_model=PyfuncClassifier(pipeline), 
    conda_env=conda_env,
    artifacts={'label_path': label_path}
    )
  
  # retrieve run ID to attach topic name later
  classification_py_id = mlflow.active_run().info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Complaint dispute prediction
# MAGIC We'd like to classify whether the complainant is 'disputed'. Using some labelled data, we will extrapolate this field for all complaints based on the complaint narrative. Note that the goal will be to extract a probability of a complaint dispute as a proxy for complaint severity rather than a binary classification (disputed or not)

# COMMAND ----------

# DBTITLE 1,Access labelled complaints
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
import re

@udf("string")
def clean_udf(s):
  s = s.lower()
  s = re.sub("[^a-z]", " ", s)
  s = re.sub("\\s+", " ", s)
  return s

corpus = spark \
  .read \
  .table("antoine_fsi.complaints_anonymized") \
  .filter(F.col("consumer_disputed").isin([0, 1])) \
  .select(
    clean_udf(F.col("complaint")).alias("complaint"), 
    F.col("consumer_disputed").alias("label")
  )

document_df = corpus.toPandas()

# COMMAND ----------

# DBTITLE 1,Balance classes
from sklearn.utils import resample
import pandas as pd
import numpy as np

df_majority = document_df[document_df['label'] == 0]
df_minority = document_df[document_df['label'] == 1]

df_negative = resample(
    df_majority, 
    replace=True,                   # sample with replacement
    n_samples=df_minority.shape[0], # to match majority class
    random_state=123                # reproducible results
  )    

# Combine minority class with downspampled majority class
df_sampled = pd.concat([df_negative, df_minority])
df_sampled['label'].value_counts().plot(kind='bar')

# COMMAND ----------

# DBTITLE 1,Split train and test
from sklearn.model_selection import train_test_split

y = df_sampled['label']
X = df_sampled['complaint']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("Training set contains {} records".format(X_train.shape[0]))
print("Testing set contains {} records".format(X_test.shape[0]))

# COMMAND ----------

# DBTITLE 1,Predict distressed complaints
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import TransformerMixin, BaseEstimator, clone
import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name='complaint_distressed_sklearn'):

  # get mlflow run Id
  run_id = mlflow.active_run().info.run_id
  
  # Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer
  count_vect = CountVectorizer(ngram_range=(3,3))
  tf_transformer = TfidfTransformer(use_idf=True)
  clf = LogisticRegression('l2', tol=0.0001, C=1.0)

  # define pipeline
  pipeline = Pipeline([
    ('vect', count_vect),
    ('tfidf', tf_transformer),
    ('clf', clf)
  ])

  # Train pipeline
  pipeline.fit(X_train, y_train)  
  y_pred = pipeline.predict(X_test)
  accuracy = accuracy_score(y_pred, y_test)
  
  # Log pipeline to mlflow
  mlflow.sklearn.log_model(pipeline, "pipeline")
  mlflow.log_metric("accuracy", accuracy)
  
  # Log classification report
  clsf_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
  clsf_report.to_csv('distressed_report.csv', index=True)
  mlflow.log_artifact('distressed_report.csv')

clsf_report

# COMMAND ----------

# MAGIC %md
# MAGIC Once again, we do have a fully functional model (in spite of its apparent bad accuracy), it expects a numpy array of text as an input and returns a simple prediction as output. In real life scenario, one may interact with model from a dataframe (having a column containing text) and expect a probability as a proxy of severity as an output. Therefore, we will repackage our sklearn pipeline as a `pyfunc` object to support this extra business logic.

# COMMAND ----------

# DBTITLE 1,Package pipeline as a pyfunc
class PyfuncSeverity(mlflow.pyfunc.PythonModel):
  
  import re
  def _clean(self, s):
    s = s.lower()
    s = re.sub("[^a-z]", " ", s)
    s = re.sub("\\s+", " ", s)
    return s
  
  def __init__(self, pipeline):
    self.pipeline = pipeline
    
  def predict(self, context, df):
    
    # We expect a single column dataframe as an input
    X = df[df.columns[0]].map(lambda x: self._clean(x))
    probs = self.pipeline.predict_proba(X)
    
    # return probability of complaint being a distressed call
    y = probs[:,1]
    return y
  
# We ensure that pyfunc has registered sklearn as dependency
conda_env = mlflow.pyfunc.get_default_conda_env()
conda_env['dependencies'][2]['pip'] += ['scikit-learn']

# We register our uber model as another mlflow experiment
with mlflow.start_run(run_name='complaint_severity_pyfunc'):
  
  # log model
  mlflow.pyfunc.log_model(
    'pipeline', 
    python_model=PyfuncSeverity(pipeline), 
    conda_env=conda_env
    )
  
  # retrieve run ID to attach topic name later
  severity_py_id = mlflow.active_run().info.run_id

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model serving and operation data
# MAGIC With all experiments logged in one place, data scientists can easily find the best model fit, enabling operation teams to retrieve the approved model (as part of their model risk management process) and surface those insights to end users or downstream processes, shortening model lifecycle processes from months to weeks. 

# COMMAND ----------

# DBTITLE 1,Publish models to ML registry
import mlflow
client = mlflow.tracking.MlflowClient()

mlflow.register_model(
    "runs:/{}/pipeline".format(classification_py_id),
    "complaint_classification"
)

mlflow.register_model(
    "runs:/{}/pipeline".format(severity_py_id),
    "complaint_severity"
)

# COMMAND ----------

# DBTITLE 1,Apply model to operation data flow (data increment)
from pyspark.sql import functions as F
from pyspark.sql.types import *
import mlflow
import mlflow.pyfunc

# load our model as a spark UDF
classification_udf = mlflow.pyfunc.spark_udf(spark, "models:/complaint_classification/production", StringType())
severity_udf = mlflow.pyfunc.spark_udf(spark, "models:/complaint_severity/production")

# Reading as a stream, processing record since last check point
# Reading from a delta table (the operation table, not the anonymized)
input_stream = spark \
  .readStream \
  .format("delta") \
  .table("complaints.complaints_bronze")
  
# Enrich incoming complaints as they unfold
output_stream = input_stream \
    .withColumn("ai_product", classification_udf("complaint")) \
    .withColumn("ai_dispute", severity_udf("complaint"))
  
# Create a streaming job triggered only once that only processes data since last checkpoint
# Write enriched records to delta table for BI reports and downstream operations
output_stream \
  .writeStream \
  .trigger(once=True) \
  .option("checkpointLocation", "/tmp/complaints_checkpoint") \
  .format("delta") \
  .table("complaints.complaints_gold")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/01_rep_etl.html">STAGE1</a>: Using Delta Lake for ingesting anonymized customer complaints in real time
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/02_rep_eda.html">STAGE2</a>: Exploring complaints data at scale using Koalas and Pandas
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/03_rep_modelling.html">STAGE3</a>: Leverage AI to better operate customer complaints
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/04_rep_augmented.html">STAGE4</a>: Supercharge your BI reports with augmented intelligence
# MAGIC ---
