# Databricks notebook source
# MAGIC %md
# MAGIC %md 
# MAGIC # Reputation risk - Exploration at scale
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
# MAGIC In order to better understand data, data scientists would traditionally sample large data sets into smaller sets that they can explore more in depth (sometimes on their laptops) using tools they are familiar with, such as Pandas dataframe and matplotlib visualisations. In order to minimize data movement across platforms (therefore minimizing the risk associated with moving data) and maximize the efficiency and effectiveness of exploratory data analysis at scale, we demonstrate how Koalas API can be used to explore all of your data with a syntax data scientists are most familiar with (similar to Pandas). 
# MAGIC 
# MAGIC ### Dependencies
# MAGIC As reported in below cell, we use multiple 3rd party libraries that must be made available across Spark cluster. Assuming you are running this notebook on a Databricks cluster that does not make use of the ML runtime, you can use `dbutils.library.installPyPI()` utility to install python libraries in that specific notebook context. For java based libraries, or if you are using an 7.X ML runtime, please follow these [alternative steps](https://docs.databricks.com/libraries.html#workspace-library) to load libraries to your environment.
# MAGIC Need DBR 7.x with the following deps installed

# COMMAND ----------

# shashank
!pip install koalas
!pip install wordcloud
# shashank

# COMMAND ----------

# # shashank
# !pip install koalas==1.2.0
# !pip install wordcloud==1.8.0
# # shashank

# COMMAND ----------

# DBTITLE 1,Install libraries
# dbutils.library.installPyPI("koalas", "1.2.0")
# dbutils.library.installPyPI("wordcloud", "1.8.0")
# dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Convert pyspark to Koalas
# import databricks.koalas as ks
kdf = spark.read.table("complaints.complaints_bronze_anonymized").to_koalas()
kdf.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC In the below example we explore all of J.P Morgan Chase complaints using simple pandas like syntax while still utilizing the distributed spark engine under the hood. 

# COMMAND ----------

# DBTITLE 1,Explore complaints products for JP Morgan Chase
# syntax is identical to Pandas dataframe
jp_kdf = kdf[kdf['company'] == 'JPMORGAN CHASE & CO.']
jp_kdf['product'].value_counts().head(18).plot('bar')

# COMMAND ----------

# MAGIC %md
# MAGIC To take the analysis further, we can run term frequency analysis on customer complaints to identify the top issues that were reported by customers across all the products for a particular FSI. At a glance we can easily identify issues related to victim identity theft and unfair debt collection.

# COMMAND ----------

# DBTITLE 1,Compute ngram frequencies with Spark ML API
from pyspark.sql import functions as F
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import NGram
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
import re

def get_ngrams_tf(df, n, total):
  """
  Given a corpus dataframe, we tokenize content, remove stop words and compute term frequency for ngrams
  We return the top ngrams as a dictionary
  """

  tokenizer = RegexTokenizer(inputCol="complaint", outputCol="words", pattern="\\W")
  tokenized = tokenizer.transform(df)

  remover = StopWordsRemover(inputCol="words", outputCol="filtered")
  remover.setStopWords(remover.getStopWords() + ['9999', 'jp', 'morgan', 'chase', '00'])
  removed = remover.transform(tokenized)

  ngramer = NGram(n=n, inputCol="filtered", outputCol="ngrams")
  ngramed = ngramer.transform(removed)

  count_df = ngramed \
    .select(F.explode(F.col("ngrams")).alias("ngram")) \
    .groupBy("ngram").count() \
    .orderBy(F.desc("count")) \
    .limit(total) \
    .toPandas()
  
  return dict(zip(count_df['ngram'], count_df['count']))

# COMMAND ----------

# DBTITLE 1,Top trigrams
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = spark.read.table("complaints.complaints_bronze_anonymized").select("complaint")
tri_df = pd.DataFrame.from_dict(get_ngrams_tf(df, 3, 20), orient='index', columns=['count'])
tri_df['trigram'] = tri_df.index

plt.figure(figsize = (20,6))
ax = sns.barplot(x= "trigram", y = "count", data = tri_df, color="steelblue", orient='v')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.xlabel('')
plt.title('Most common trigrams')
plt.ylabel('count')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We can dig in further into individual products such as Consumer Loans and Credit Cards using a word cloud to better understand what the customers are complaining about.

# COMMAND ----------

!pip install wordcloud

# COMMAND ----------

# shashank

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Get list of distinct products
products_df = spark.read.table("complaints.complaints_bronze_anonymized").select("product").distinct().toPandas()
classes = list(products_df['product'].values)

# Function to generate word cloud for a given product and n value
def word_cloud(product, n):
    df = spark.read.table("complaints.complaints_bronze_anonymized") \
        .filter(F.col("product") == product) \
        .filter(F.col("company") == 'JPMORGAN CHASE & CO.') \
        .select("complaint")

    # Check if dataframe is empty
    if df.rdd.isEmpty():
        print(f"No data found for product: {product}")
        return None

    # Generate word cloud from dataframe
    wc = WordCloud(
        random_state=42, 
        background_color="white",
        width=300,
        height=300
    ).generate_from_frequencies(get_ngrams_tf(df, n, 300))

    return wc

# Create figure and generate word clouds for each product
fig, axs = plt.subplots(len(classes)//4+1, 4, figsize=(20, 5 * (len(classes)//4+1)))
for i, clazz in enumerate(classes):
    ax = axs.flat[i]
    ax.set_title(clazz)
    wordcloud = word_cloud(clazz, 1)

    # Check if word cloud was generated successfully
    if wordcloud is not None:
        ax.imshow(wordcloud)
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, "No data found", ha="center", va="center")

# Set overall title for the figure
fig.suptitle("Word Clouds for Products")

plt.show()


# COMMAND ----------

# DBTITLE 1,Top ngrams for each product at JP Morgan Chase
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# classes = list(spark.read.table("complaints.complaints_bronze_anonymized").select("product").distinct().toPandas()['product'].values)

# def word_cloud(product, n):
  
#   df = spark.read.table("complaints.complaints_bronze_anonymized") \
#     .filter(F.col("product") == product) \
#     .filter(F.col("company") == 'JPMORGAN CHASE & CO.') \
#     .select("complaint")
  
#   return WordCloud(
#     random_state=42, 
#     background_color="white",
#     width=300,
#     height=300
#   ).generate_from_frequencies(get_ngrams_tf(df, n, 300))

# fig = plt.figure(figsize=(20, 20))
# for i, clazz in enumerate(classes):
#   ax = fig.add_subplot(len(classes), 4, i + 1)
#   ax.set_title(clazz)
#   wordcloud = word_cloud(clazz, 1)
#   ax.imshow(wordcloud)
#   ax.axis('off')
  
# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Complaint classification
# MAGIC In order to validate the predictive potential of our consumer data and therefore confirm our dataset is a great fit for ML, we can identify similarity between complaints by using t-Distributed Stochastic Neighbor Embedding (t-SNE) as per below example. For that purpose, we will first train a [word2vec](https://spark.apache.org/docs/latest/ml-features.html#word2vec) model to learn semantic value of each complaint.

# COMMAND ----------

# DBTITLE 1,Learn complaints vocabulary
# MAGIC %scala
# MAGIC import org.apache.spark.ml.feature.RegexTokenizer
# MAGIC import org.apache.spark.ml.feature.Word2Vec
# MAGIC import org.apache.spark.ml.linalg.Vector
# MAGIC import org.apache.spark.ml.{Pipeline, PipelineModel}
# MAGIC import org.apache.spark.sql.Row
# MAGIC import org.apache.spark.sql.functions._
# MAGIC 
# MAGIC val df = spark.read.table("complaints.complaints_bronze_anonymized").select("complaint_id", "company", "product", "complaint")
# MAGIC val tokenizer = new RegexTokenizer().setInputCol("complaint").setOutputCol("words").setPattern("\\W")
# MAGIC val word2vec = new Word2Vec().setInputCol("words").setOutputCol("features").setVectorSize(255)
# MAGIC val pipeline = new Pipeline().setStages(Array(tokenizer, word2vec))
# MAGIC val pipelineModel = pipeline.fit(df)

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC val to_array = udf((v: Vector) => v.toArray)
# MAGIC val ratio = udf((count: Long) => 2500.0 / count)
# MAGIC val ratios = df.groupBy("product").count().withColumn("ratio", ratio(col("count"))).rdd.map(r => {
# MAGIC   (r.getAs[String]("product"), r.getAs[Double]("ratio"))
# MAGIC }).collectAsMap().toMap
# MAGIC 
# MAGIC val keyCount = df.rdd.keyBy(x => x.getAs[String]("product")).countByKey()
# MAGIC val maxRatio = keyCount.values.max
# MAGIC val sampleFractions = keyCount.map(x => (x._1, Math.min(1.0, maxRatio.toDouble / x._2.toDouble))).toMap
# MAGIC 
# MAGIC pipelineModel
# MAGIC   .transform(df.stat.sampleBy("product", fractions = sampleFractions, seed = 42))
# MAGIC   .select("complaint_id", "product", "features")
# MAGIC   .withColumn("features", to_array(col("features")))
# MAGIC   .createOrReplaceTempView("word2vec")

# COMMAND ----------

# DBTITLE 1,Transform complaint narrative into input vectors
# %scala

# // orignal

# val to_array = udf((v: Vector) => v.toArray)
# val ratio = udf((count: Long) => 2500.0 / count)
# val ratios = df.groupBy("product").count().withColumn("ratio", ratio(col("count"))).rdd.map(r => {
#   (r.getAs[String]("product"), r.getAs[Double]("ratio"))
# }).collectAsMap().toMap

# val keyCount = df.rdd.keyBy(x => x.getAs[String]("product")).countByKey()
# val maxRatio = keyCount.values.max
# val sampleFractions = keyCount.map(x => (x._1, Math.min(2500.0, maxRatio) / x._2)).toMap

# pipelineModel
#   .transform(df.stat.sampleBy("product", fractions = ratios, seed = 42))
#   .select("complaint_id", "product", "features")
#   .withColumn("features", to_array(col("features")))
#   .createOrReplaceTempView("word2vec")

# COMMAND ----------

# DBTITLE 1,Visualize classified complaints vs. vocabulary used
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

# Access features vectors for each complaint, and convert column of array to numpy 
word2vec_df = spark.read.table("word2vec").toPandas()
tnse_input = np.array(word2vec_df['features'].values.tolist())

# Fit a t-Distributed Stochastic Neighbor Embedding (t-SNE)
tsne = TSNE(n_components=2, perplexity=500, n_iter=1000)
tsne_results = tsne.fit_transform(tnse_input)

# Attach the results of t-SNE back to original dataframe
word2vec_df['X'] = tsne_results[:,0]
word2vec_df['Y'] = tsne_results[:,1]

# Visualize complaint universe color coded by product type
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="X", y="Y",
    hue="product",
    palette=sns.color_palette("hls", 4),
    data=word2vec_df,
    legend="full",
    alpha=0.8
)

# COMMAND ----------

# MAGIC %md
# MAGIC Although some consumer complaints may overlap in terms of possible categories (both secure and unsecure lending exhibit similar keywords), we can observe distinct clusters, indicative of patterns that could easily be machine learned. The above plot re-confirms a pattern that would enable us to classify complaints. The potential overlap also indicates that some complaints could easily be misclassified by end users or agents, resulting in a suboptimal complaint management system and poor customer experience. 

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/01_rep_etl.html">STAGE1</a>: Using Delta Lake for ingesting anonymized customer complaints in real time
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/02_rep_eda.html">STAGE2</a>: Exploring complaints data at scale using Koalas and Pandas
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/03_rep_modelling.html">STAGE3</a>: Leverage AI to better operate customer complaints
# MAGIC + <a href="https://databricks.com/notebooks/reprisk_notebooks/04_rep_augmented.html">STAGE4</a>: Supercharge your BI reports with augmented intelligence
# MAGIC ---
