# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/fine-grained-demand-forecasting. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/demand-forecasting.

# COMMAND ----------

# MAGIC %md The objective of this notebook is to illustrate how we might generate a large number of fine-grained forecasts at the store-item level in an efficient manner leveraging the distributed computational power of Databricks.  This is a SparklyR rewrite of an [existing notebook](https://www.databricks.com/blog/2021/04/06/fine-grained-time-series-forecasting-at-scale-with-facebook-prophet-and-apache-spark-updated-for-spark-3.html) previous written using Python.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC For this exercise, we will make use of an increasingly popular library for demand forecasting, [Prophet](https://facebook.github.io/prophet/).
# MAGIC 
# MAGIC **NOTE** R package installation can be a slow process.  (We found the installation of the packages below took a little over 25 minutes on average.) To explore techniques for speeding up R package installs, please check out [this document](https://github.com/marygracemoesta/R-User-Guide/blob/master/Developing_on_Databricks/package_management.md). In addition, you may consider installing some packages from RStudio Package Manager using a *repos* value of *`https://packagemanager.rstudio.com/all/__linux__/focal/latest`*.

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Install Packages
# shashank
Sys.setenv(LIBARROW_BINARY = TRUE)

install.packages("arrow", repos="https://cloud.r-project.org")
install.packages("dplyr")
install.packages("prophet", repos="https://cran.rstudio.com/")
install.packages("lubridate")
install.packages("Metrics")
# shashank

# COMMAND ----------

# DBTITLE 1,Load Required Packages
# MAGIC %r
# MAGIC library(arrow)
# MAGIC library(prophet)
# MAGIC library(lubridate)
# MAGIC library(Metrics)
# MAGIC library(dplyr)
# MAGIC library(sparklyr)
# MAGIC 
# MAGIC # associate sparklyr with databricks spark instance
# MAGIC sc <- sparklyr::spark_connect(method = "databricks")

# COMMAND ----------

# MAGIC %md ## Step 1: Examine the Data
# MAGIC 
# MAGIC For our training dataset, we will make use of 5-years of store-item unit sales data for 50 items across 10 different stores.  This data set is publicly available as part of a past Kaggle competition and can be downloaded with the `./config/Data Extract` notebook with your own Kaggle credentials.
# MAGIC 
# MAGIC Once downloaded, we can unzip the *train.csv.zip* file and upload the decompressed CSV to */FileStore/demand_forecast/train/* using the file import steps documented [here](https://docs.databricks.com/data/databricks-file-system.html#!#user-interface). With the dataset accessible within Databricks, we can now explore it in preparation for modeling:

# COMMAND ----------

# MAGIC %run "./config/Data Extract"

# COMMAND ----------

# DBTITLE 1,Access the Dataset
# MAGIC %r
# MAGIC # read the training file into a spark table
# MAGIC train_data = spark_read_csv(sc, name = "train_data",  path = "/tmp/solacc/demand_forecast/train/train.csv")
# MAGIC 
# MAGIC # show data
# MAGIC show(train_data)

# COMMAND ----------

# MAGIC %md When performing demand forecasting, we are often interested in general trends and seasonality.  Let's start our exploration by examining the annual trend in unit sales:

# COMMAND ----------

# DBTITLE 1,View Yearly Trends
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   year(date) as year, 
# MAGIC   sum(sales) as sales
# MAGIC FROM train_data
# MAGIC GROUP BY year(date)
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %md It's very clear from the data that there is a generally upward trend in total unit sales across the stores. If we had better knowledge of the markets served by these stores, we might wish to identify whether there is a maximum growth capacity we'd expect to approach over the life of our forecast.  But without that knowledge and by just quickly eyeballing this dataset, it feels safe to assume that if our goal is to make a forecast a few days, months or even a year out, we might expect continued linear growth over that time span.
# MAGIC 
# MAGIC Now let's examine seasonality.  If we aggregate the data around the individual months in each year, a distinct yearly seasonal pattern is observed which seems to grow in scale with overall growth in sales:

# COMMAND ----------

# DBTITLE 1,View Monthly Trends
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') as month,
# MAGIC   SUM(sales) as sales
# MAGIC FROM train_data
# MAGIC GROUP BY TRUNC(date, 'MM')
# MAGIC ORDER BY month;

# COMMAND ----------

# MAGIC %md Aggregating the data at a weekday level, a pronounced weekly seasonal pattern is observed with a peak on Sunday (weekday 0), a hard drop on Monday (weekday 1) and then a steady pickup over the week heading back to the Sunday high.  This pattern seems to be pretty stable across the five years of observations:

# COMMAND ----------

# DBTITLE 1,View Weekday Trends
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   YEAR(date) as year,
# MAGIC   (
# MAGIC     CASE
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sun' THEN 0
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Mon' THEN 1
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Tue' THEN 2
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Wed' THEN 3
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Thu' THEN 4
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Fri' THEN 5
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sat' THEN 6
# MAGIC     END
# MAGIC   ) % 7 as weekday,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM train_data
# MAGIC   GROUP BY date
# MAGIC  ) x
# MAGIC GROUP BY year, weekday
# MAGIC ORDER BY year, weekday;

# COMMAND ----------

# MAGIC %md Now that we are oriented to the basic patterns within our data, let's explore how we might build a forecast.

# COMMAND ----------

# MAGIC %md ## Step 2: Build a Single Forecast
# MAGIC 
# MAGIC Before attempting to generate forecasts for individual combinations of stores and items, it might be helpful to build a single forecast for no other reason than to orient ourselves to the use of Prophet library.
# MAGIC 
# MAGIC Our first step is to assemble the historical dataset on which we will train the model:

# COMMAND ----------

# DBTITLE 1,Retrieve Data for a Single Item-Store Combination
# MAGIC %r
# MAGIC 
# MAGIC # query to aggregate data to date (ds) level
# MAGIC sql_statement = "
# MAGIC   SELECT
# MAGIC     CAST(date as date) as ds,
# MAGIC     sales as y
# MAGIC   FROM train_data
# MAGIC   WHERE store=1 AND item=1
# MAGIC   ORDER BY ds
# MAGIC   "
# MAGIC 
# MAGIC # assemble dataset in R dataframe
# MAGIC history_df = collect(DBI::dbGetQuery(sc , sql_statement))
# MAGIC 
# MAGIC # drop any missing records
# MAGIC history_df <- history_df[complete.cases(history_df), ]

# COMMAND ----------

# MAGIC %md Based on our review of the data, it looks like we should set our overall growth pattern to linear and enable the evaluation of weekly and yearly seasonal patterns. We might also wish to set our seasonality mode to multiplicative as the seasonal pattern seems to grow with overall growth in sales:

# COMMAND ----------

# DBTITLE 1,Train Prophet Model
# MAGIC %r
# MAGIC # set model parameters and fit the model to historical data
# MAGIC model = prophet(history_df,
# MAGIC   interval.width=0.95,
# MAGIC   growth='linear',
# MAGIC   daily.seasonality=FALSE,
# MAGIC   weekly.seasonality=TRUE,
# MAGIC   yearly.seasonality=TRUE,
# MAGIC   seasonality.mode='multiplicative'
# MAGIC   )

# COMMAND ----------

# MAGIC %md Now that we have a trained model, let's use it to build a 90-day forecast:

# COMMAND ----------

# DBTITLE 1,Build Forecast
# MAGIC %r
# MAGIC #Build forecast
# MAGIC # define a dataset including both historical dates & 90-days beyond the last available date
# MAGIC future_df = make_future_dataframe(
# MAGIC   model,
# MAGIC   periods=90, 
# MAGIC   freq='days', 
# MAGIC   include_history=TRUE
# MAGIC   )
# MAGIC 
# MAGIC # predict over the dataset
# MAGIC forecast_df = predict(model, future_df)
# MAGIC 
# MAGIC display(forecast_df)

# COMMAND ----------

# MAGIC %md How did our model perform? Here we can see the general and seasonal trends in our model presented as graphs:

# COMMAND ----------

# DBTITLE 1,Examine Forecast Components
# MAGIC %r
# MAGIC prophet_plot_components(model, forecast_df)

# COMMAND ----------

# MAGIC %md And here, we can see how our actual and predicted data line up as well as a forecast for the future:

# COMMAND ----------

# DBTITLE 1,View Historicals vs. Predictions
# MAGIC %r
# MAGIC # adjust model history for plotting purposes
# MAGIC model$history <- dplyr::filter(model$history, model$history$ds > ymd("2017-01-01"))
# MAGIC # plot history and forecast for relevant period
# MAGIC plot(
# MAGIC   model,
# MAGIC   dplyr::filter(forecast_df, forecast_df$ds > ymd("2017-01-01")),
# MAGIC   xlabel='date',
# MAGIC   ylabel='sales'
# MAGIC   )

# COMMAND ----------

# MAGIC %md **NOTE** This visualization is a bit busy. Bartosz Mikulski provides [an excellent breakdown](https://www.mikulskibartosz.name/prophet-plot-explained/) of it that is well worth checking out.  In a nutshell, the black dots represent our actuals with the darker blue line representing our predictions and the lighter blue band representing our (95%) uncertainty interval.

# COMMAND ----------

# MAGIC %md Visual inspection is useful, but a better way to evaluate the forecast is to calculate Mean Absolute Error, Mean Squared Error and Root Mean Squared Error values for the predicted relative to the actual values in our set:

# COMMAND ----------

# DBTITLE 1,Calculate Evaluation metrics
# MAGIC %r
# MAGIC # get historical actuals & predictions for comparison
# MAGIC actuals_df = subset(history_df, ds < ymd("2018-01-01"), select = "y")
# MAGIC predicted_df = subset(forecast_df, ds < ymd("2018-01-01"), select = "yhat")
# MAGIC 
# MAGIC # calculate evaluation metrics
# MAGIC mae = mae(actuals_df$y,predicted_df$yhat)
# MAGIC mse = mse(actuals_df$y,predicted_df$yhat)
# MAGIC rmse = sqrt(mse)
# MAGIC 
# MAGIC # print metrics to the screen
# MAGIC writeLines( paste("\n MAE:", mae , "\n MSE:", mse , "\n RMSE:", rmse) )

# COMMAND ----------

# MAGIC %md Prophet provides [additional means](https://facebook.github.io/prophet/docs/diagnostics.html) for evaluating how your forecasts hold up over time. You're strongly encouraged to consider using these and those additional techniques when building your forecast models but we'll skip this here to focus on the scaling challenge.

# COMMAND ----------

# MAGIC %md ## Step 3: Scale Forecast Generation
# MAGIC 
# MAGIC With the mechanics under our belt, let's now tackle our original goal of building numerous, fine-grain models & forecasts for individual store and item combinations.  We will start by assembling sales data at the store-item-date level of granularity:
# MAGIC 
# MAGIC **NOTE**: The data in this data set should already be aggregated at this level of granularity but we are explicitly aggregating to ensure we have the expected data structure.

# COMMAND ----------

# DBTITLE 1,Retrieve Data for All Store-Item Combinations
# MAGIC %r
# MAGIC sql_statement = "
# MAGIC   SELECT
# MAGIC     store,
# MAGIC     item,
# MAGIC     CAST(date as date) as ds,
# MAGIC     SUM(sales) as y
# MAGIC   FROM train_data
# MAGIC   GROUP BY store, item, ds
# MAGIC   ORDER BY store, item, ds
# MAGIC   "
# MAGIC 
# MAGIC store_item_history = sdf_sql( sc , sql_statement)

# COMMAND ----------

# MAGIC %md With our data aggregated at the store-item-date level, we need to consider how we will pass our data to FBProphet. If our goal is to build a model for each store and item combination, we will need to pass in a store-item subset from the dataset we just assembled, train a model on that subset, and receive a store-item forecast back. We'd expect that forecast to be returned as a dataset with a structure like this where we retain the store and item identifiers for which the forecast was assembled and we limit the output to just the relevant subset of fields generated by the Prophet model:

# COMMAND ----------

# DBTITLE 1,Define Schema for Forecast Output
# MAGIC %r
# MAGIC result_schema <- list( 
# MAGIC                        store = "integer",
# MAGIC                        item = "integer",
# MAGIC                        ds = "date",
# MAGIC                        y = "integer",
# MAGIC                        yhat= "double", 
# MAGIC                        yhat_upper = "double", 
# MAGIC                        yhat_lower = "double"
# MAGIC                       )

# COMMAND ----------

# MAGIC %md To train the model and generate we will leverage [*spark_apply*](https://spark.rstudio.com/guides/distributed-r). The spark_apply function groups the data in a dataframe and applies a function to each grouping. 
# MAGIC 
# MAGIC For the function to be applied to each grouping, we will write a custom function which will train a model and generate a forecast much like what was done previously in this notebook.

# COMMAND ----------

# DBTITLE 1,Define Function to Train Model & Generate Forecast
# MAGIC %r
# MAGIC  forecast_store_item <- function( history_df ) {
# MAGIC    
# MAGIC   # Load prophet package
# MAGIC   library(prophet)
# MAGIC   library(dplyr)
# MAGIC   
# MAGIC   # TRAIN MODEL AS BEFORE
# MAGIC   # --------------------------------------
# MAGIC   # remove missing values (more likely at day-store-item level)
# MAGIC   history_df <- history_df[complete.cases(history_df), ]
# MAGIC 
# MAGIC   # set model parameters and fit the model to historical data
# MAGIC     model = prophet(
# MAGIC       history_df,
# MAGIC       interval.width=0.95,
# MAGIC       growth='linear',
# MAGIC       daily.seasonality=FALSE,
# MAGIC       weekly.seasonality=TRUE,
# MAGIC       yearly.seasonality=TRUE,
# MAGIC       seasonality.mode='multiplicative'
# MAGIC     )
# MAGIC   # --------------------------------------
# MAGIC   
# MAGIC   # BUILD FORECAST AS BEFORE
# MAGIC   # --------------------------------------
# MAGIC   # define a dataset including both historical dates & 90-days beyond the last available date
# MAGIC   future_df = make_future_dataframe(
# MAGIC     model,
# MAGIC     periods=90, 
# MAGIC     freq='days', 
# MAGIC     include_history=TRUE
# MAGIC    )
# MAGIC   
# MAGIC   # make predictions
# MAGIC   # predict over the dataset
# MAGIC   forecast_df = predict(model, future_df)
# MAGIC   # --------------------------------------
# MAGIC    
# MAGIC   # ASSEMBLE EXPECTED RESULT SET
# MAGIC   # --------------------------------------
# MAGIC    
# MAGIC   # get relevant fields from forecast
# MAGIC   f_df <- forecast_df %>% select(ds, yhat, yhat_upper, yhat_lower)
# MAGIC   # convert date
# MAGIC   f_df['ds'] = as.Date(as.character(as.POSIXct(f_df[["ds"]])))  
# MAGIC   
# MAGIC   # get relevant fields from history
# MAGIC   h_df <- history_df %>% select(ds, y)
# MAGIC    
# MAGIC   # join history and forecast
# MAGIC   results_df <- merge(h_df, f_df, by = "ds", all = TRUE)
# MAGIC 
# MAGIC   # return expected dataset
# MAGIC   return(results_df)
# MAGIC                     
# MAGIC  }

# COMMAND ----------

# MAGIC %md There's a lot taking place within our function, but if you compare the first two blocks of code within which the model is being trained and a forecast is being built to the cells in the previous portion of this notebook, you'll see the code is pretty much the same as before. It's only in the assembly of the required result set that truly new code is being introduced and it consists of fairly standard R dataframe manipulations.

# COMMAND ----------

# MAGIC %md Now let's call our spark_apply function to build our forecasts.  We do this by grouping our historical dataset around store and item as we pass these as keys.  We then apply our function to each group.
# MAGIC 
# MAGIC **Note:** To get the best performance, we specify the schema of the expected output DataFrame which we defined earlier as *result_schema*  to *spark_apply* with the *columns* argument.  This is optional, but if we don't supply the schema Spark will need to sample the output to infer it.  This can be quite costly on longer running UDFs.

# COMMAND ----------

# DBTITLE 1,Apply Forecast Function to Each Store-Item Combination
# MAGIC %r
# MAGIC results <- spark_apply(store_item_history, forecast_store_item, columns = result_schema, group_by = (c('store', 'item')))  
# MAGIC 
# MAGIC # add model training date
# MAGIC results <- mutate(results, training_date = current_date())
# MAGIC 
# MAGIC show(results)

# COMMAND ----------

# MAGIC %md We we are likely wanting to report on our forecasts, so let's save them to a queryable table structure:
# MAGIC 
# MAGIC **NOTE** Using *append* mode with *spark_write_table* requires the target table to exist which is why we've inserted a *CREATE TABLE* statement into the code block. 

# COMMAND ----------

# DBTITLE 1,Persist Forecast Output
# MAGIC %r
# MAGIC forecasts <- results %>% select(date = ds, store, item, sales = y, sales_predicted = yhat, sales_predicted_upper = yhat_upper, sales_predicted_lower = yhat_lower, training_date)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS 
# MAGIC forecasts_sparklyr (
# MAGIC   date DATE ,
# MAGIC   store INT,
# MAGIC   item INT ,
# MAGIC   sales INT,
# MAGIC   sales_predicted DOUBLE ,
# MAGIC   sales_predicted_upper DOUBLE ,
# MAGIC   sales_predicted_lower DOUBLE,
# MAGIC   training_date DATE 
# MAGIC   )

# COMMAND ----------

# MAGIC %r
# MAGIC spark_write_table(forecasts, name = "forecasts_sparklyr", mode = "append") 

# COMMAND ----------

# MAGIC %md But how good (or bad) is each forecast?  Using the pandas function technique, we can generate evaluation metrics for each store-item forecast as follows:

# COMMAND ----------

# shashank

# Load sparklyr package
library(sparklyr)
library(lubridate)
library(dplyr)

# shashank

# COMMAND ----------

# DBTITLE 1,Apply Same Techniques to Evaluate Each Forecast
# MAGIC %r
# MAGIC # schema of expected result set
# MAGIC eval_schema = list(                       
# MAGIC                     training_date = "date",
# MAGIC                     store = "integer",
# MAGIC                     item = "integer",
# MAGIC                     mae = "double",
# MAGIC                     mse = "double",
# MAGIC                     rmse = "double"
# MAGIC   )
# MAGIC 
# MAGIC # define function to calculate metrics
# MAGIC evaluate_forecast <- function( evaluation_df ) {
# MAGIC   
# MAGIC   #load package
# MAGIC   library(Metrics)
# MAGIC   
# MAGIC   # calculate evaluation metrics
# MAGIC   mae = mae( evaluation_df$sales, evaluation_df$sales_predicted )
# MAGIC   mse = mse( evaluation_df$sales, evaluation_df$sales_predicted )
# MAGIC   rmse = sqrt( mse )
# MAGIC   
# MAGIC   # assemble result set
# MAGIC   results = data.frame( mae = mae, mse = mse, rmse = rmse)
# MAGIC   
# MAGIC   return( results )
# MAGIC   }
# MAGIC  
# MAGIC # calculate metrics for each store-item combination
# MAGIC 
# MAGIC forecasts <- filter(forecasts, training_date==current_date())
# MAGIC 
# MAGIC # orignal
# MAGIC # eval_results <- spark_apply(filter(forecasts, date<lubridate::ymd("2018-01-01")), evaluate_forecast, columns = eval_schema, group_by = (c('training_date','store', 'item'))) 
# MAGIC 
# MAGIC # shashank
# MAGIC eval_results <- spark_apply(filter(forecasts, date < to_date('2018-01-01')), evaluate_forecast, columns = eval_schema, group_by = c('training_date', 'store', 'item'))
# MAGIC # shashank

# COMMAND ----------

# MAGIC %md Once again, we will likely want to report the metrics for each forecast, so we persist these to a queryable table:

# COMMAND ----------

# DBTITLE 1,Persist Evaluation Metrics
# MAGIC %r
# MAGIC forecast_evals <- eval_results %>% select(store, item, mae, mse, rmse, training_date)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS 
# MAGIC forecast_evals_sparklyr (
# MAGIC   store INT,
# MAGIC   item INT ,
# MAGIC   mae DOUBLE ,
# MAGIC   mse DOUBLE ,
# MAGIC   rmse DOUBLE ,
# MAGIC   training_date DATE 
# MAGIC   )

# COMMAND ----------

# MAGIC %r
# MAGIC spark_write_table(forecast_evals, name = "forecast_evals_sparklyr", mode = "append") 

# COMMAND ----------

# MAGIC %md We now have constructed a forecast for each store-item combination and generated basic evaluation metrics for each. To see this forecast data, we can issue a simple query (limited here to product 1 across stores 1 through 3):

# COMMAND ----------

# DBTITLE 1,Visualize Forecasts
# MAGIC %sql
# MAGIC  
# MAGIC SELECT
# MAGIC   store,
# MAGIC   date,
# MAGIC   sales_predicted,
# MAGIC   sales_predicted_upper,
# MAGIC   sales_predicted_lower
# MAGIC FROM forecasts_sparklyr a
# MAGIC WHERE item = 1 AND
# MAGIC       store IN (1, 2, 3) AND
# MAGIC       date >= '2018-01-01' AND
# MAGIC       training_date=current_date()
# MAGIC ORDER BY store

# COMMAND ----------

# MAGIC %md And for each of these, we can retrieve a measure of help us assess the reliability of each forecast:

# COMMAND ----------

# DBTITLE 1,Retrieve Evaluation Metrics
# MAGIC %sql
# MAGIC  
# MAGIC SELECT
# MAGIC   store,
# MAGIC   mae,
# MAGIC   mse,
# MAGIC   rmse
# MAGIC FROM forecast_evals_sparklyr a
# MAGIC WHERE item = 1 AND
# MAGIC       training_date=current_date()
# MAGIC ORDER BY store

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | prophet                                  |Implements a procedure for forecasting time series data based on an additive model |  MIT   | https://cran.r-project.org/web/packages/prophet/index.html                 |
# MAGIC | Metrics | An implementation of evaluation metrics in R that are commonly used in supervised machine learning | 	BSD 3 | https://cran.r-project.org/web/packages/Metrics/index.html | 
# MAGIC | Arrow |This package provides an interface to the 'Arrow C++' library|Apache|https://cran.r-project.org/web/packages/arrow/index.html| 
# MAGIC | lubridate |Lubridate provides tools that make it easier to parse and manipulate dates|GPL-2|https://cran.r-project.org/web/packages/lubridate/index.html| 
# MAGIC | dplyr |A fast, consistent tool for working with data frame like objects, both in memory and out of memory|MIT|https://cran.r-project.org/web/packages/dplyr/index.html| 
