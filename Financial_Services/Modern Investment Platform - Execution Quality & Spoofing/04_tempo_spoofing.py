# Databricks notebook source
# MAGIC %md
# MAGIC # Modernizing Investment Data Platforms - surveillance
# MAGIC 
# MAGIC The appetite for investment in a range of asset classes is at a historic high in 2020. As of July 2020, "Retail traders make up nearly 25% of the stock market following COVID-driven volatility" ([source](https://markets.businessinsider.com/news/stocks/retail-investors-quarter-of-stock-market-coronavirus-volatility-trading-citadel-2020-7-1029382035)). As investors gain access and trade alternative assets such as cryptocurrency, trading volumes have skyrocketed and created new data challenges. Moreover, cutting edge research is no longer restricted to institutional investors on Wall Street - today’s world of investing extends to digital exchanges in Silicon Valley, data-centric market makers, and retail brokers that are investing increasingly in AI-powered tools for investors. Data lakes have become standard for building financial data products and research, but the lack of blueprints for how to build an enterprise data lake in the cloud lakes hinder the adoption to scalable AI (such as volatility forecasting). Through a series of design patterns and real world examples, we address these core challenges and show benchmarks to illustrate why building investment data platforms on Databricks leads to cost savings and improved financial products. We also introduce [tempo](https://github.com/databrickslabs/tempo), an open source library for manipulating time series at scale.
# MAGIC 
# MAGIC ---
# MAGIC + <a href="$./01_tempo_context">STAGE0</a>: Home page
# MAGIC + <a href="$./02_tempo_etl">STAGE1</a>: Design pattern for ingesting BPipe data
# MAGIC + <a href="$./03_tempo_volatility">STAGE2</a>: Introducing tempo for calculating market volatility
# MAGIC + <a href="$./04_tempo_spoofing">STAGE3</a>: Introducing tempo for market surveillance
# MAGIC ---
# MAGIC <ricardo.portilla@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Execution Quality and Spoofing
# MAGIC Market surveillance is an important part of the financial services ecosystem which aims to reduce market manipulation, increase transparency, and enforce some baseline rules for various assets classes. Some examples of organizations, both governmental and private, which have broad surveillance programs in place include NASDAQ, FINRA, CFTC, and CME Group. As the retail investing industry gets larger with newer and inexperienced investors (source), especially in the digital currency space, it is important to understand how to build a basic surveillance program that reduces financial fraud and increases transparency in areas such as market volatility, risk, and best execution. We show here how to build basic price improvement summaries as well as putting a basic spoofing implementation together.
# MAGIC 
# MAGIC ### Dependencies
# MAGIC This notebook shows a real life example using [Tempo](https://github.com/databrickslabs/tempo), an open source library developed by DataBricks labs for time series manipulation at massive scale. Although the library can be packaged as an archive, we show how analysts could quickly install scope libraries using the `%pip` magic command. 

# COMMAND ----------

# DBTITLE 1,Install Databricks Labs tempo Package
# MAGIC %pip install git+https://github.com/databrickslabs/tempo.git

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### `STEP0` - Upload FINRA Market Participants to the FileStore
# MAGIC 
# MAGIC Please follow these [steps](https://docs.databricks.com/data/data.html#import-data-1) to upload your own market participants data to the Filestore.
# MAGIC See [Market Participant ID List](https://otce.finra.org/otce/assets/download.html) for more details.

# COMMAND ----------

# DBTITLE 1,Retrieve market participants data
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

firms = (
  spark.read.format("csv")
    .option("header", "true")
    .option("sep", "|").load("/FileStore/tables/firms.txt")
    .withColumn("rownum", row_number().over(Window.partitionBy(lit(1)).orderBy("MPID")))
)

display(firms)

# COMMAND ----------

# MAGIC %md
# MAGIC We can simulate the order information on all executions so we can calculate slippage (using an interval window of 2 seconds). In real world, this dataset represented here (`tempo.delta_tick_trades_6mo`) would be ingested directly from your source systems and can be joined with Market Participant based on `MPID`

# COMMAND ----------

# DBTITLE 1,Simulate Order / Firm Information for Exchange Trades for 6 Month Period
executions = (
  spark.table("tempo.delta_tick_trades_6mo")
    .withColumn("order_arrival_ts", expr("event_ts - interval 2 seconds"))
    .withColumn("rownum", col("size")%1102).join(firms, ["rownum"])
    .withColumn("side_cd", when(round(col("datetime"))%2 == 0, "B").otherwise("S"))
)

executions.createOrReplaceTempView("exchange_trades")

display(executions)

# COMMAND ----------

# DBTITLE 1,Broker-Dealer Trade Volume By Month 
# MAGIC %sql 
# MAGIC SELECT 
# MAGIC   month(event_ts) month, 
# MAGIC   sum(size) agg_quantity, 
# MAGIC   marketParticipantName, 
# MAGIC   count(1) trade_ct
# MAGIC FROM exchange_trades 
# MAGIC GROUP BY month(event_ts), marketParticipantName
# MAGIC ORDER BY month

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### `STEP1` - AS-OF Join using tempo
# MAGIC Using plain SQL, we can better understand the complexity of data at hand. As trades for specific tickers may vastly outnumber least popular stocks, this imbalanced dataset could theoretically impact performance of any time series analytics such as a AS-OF join. Luckily, `tempo` gives users the possibility to partition data by specific columns in order to overcome this challenge of scale and ensure an even distribution of compute power.

# COMMAND ----------

# DBTITLE 1,AS OF Join to Calculate Bid/Ask AS OF Arrival Time and AS OF Execution Time
from tempo.tsdf import TSDF

trades = spark.table("exchange_trades")
quotes = spark.table("ricardo.delta_tick_quotes_6mo")

trades_tsdf = TSDF(trades, ts_col = 'event_ts', partition_cols = ["DATE", "TICKER"])
quotes_tsdf = TSDF(quotes, ts_col='event_ts', partition_cols = ["DATE", "TICKER"])
ex_asof = trades_tsdf.asofJoin(quotes_tsdf, right_prefix = "asof_ex_time")

orders_tsdf = TSDF(ex_asof.df, ts_col = 'order_arrival_ts', partition_cols = ["DATE", "TICKER"])
order_asof = ex_asof.asofJoin(quotes_tsdf, right_prefix = "asof_ord_time")
order_asof.df.write.format("delta").mode('overwrite').saveAsTable("tempo.silver_trade_slippage")

# COMMAND ----------

# MAGIC %md
# MAGIC With our ability to join both trades and quotes data in a performant and efficient manner, we show below how this data truly acts as a data asset used for investigation and reporting using simple SQL analytics.

# COMMAND ----------

# DBTITLE 1,Slippage Results by Broker Dealer
# MAGIC %sql 
# MAGIC 
# MAGIC WITH slippages AS (
# MAGIC   SELECT
# MAGIC     *, 
# MAGIC     CASE
# MAGIC       WHEN s.side_cd = 'B' THEN s.asof_ord_time_ask_price - s.asof_ex_time_ask_price 
# MAGIC       ELSE s.asof_ex_time_bid_price - s.asof_ord_time_bid_price 
# MAGIC     END AS slippage 
# MAGIC   FROM tempo.silver_trade_slippage s
# MAGIC )
# MAGIC 
# MAGIC SELECT
# MAGIC   s.mpid, 
# MAGIC   sign(s.slippage), 
# MAGIC   log(-sum(s.slippage)) agg_slippage
# MAGIC FROM slippages s
# MAGIC WHERE sign(s.slippage) = -1
# MAGIC GROUP BY s.mpid, sign(s.slippage)
# MAGIC ORDER BY log(-sum(s.slippage)) DESC
# MAGIC LIMIT 20

# COMMAND ----------

# DBTITLE 1,Slippage by Symbol 
# MAGIC %sql 
# MAGIC 
# MAGIC WITH slippages AS (
# MAGIC   SELECT 
# MAGIC     *, 
# MAGIC     CASE
# MAGIC       WHEN s.side_cd = 'B' THEN s.asof_ord_time_ask_price - s.asof_ex_time_ask_price 
# MAGIC       ELSE s.asof_ex_time_bid_price - s.asof_ord_time_bid_price
# MAGIC     END AS slippage 
# MAGIC   FROM tempo.silver_trade_slippage s
# MAGIC   WHERE s.mpid in ('WBLR', 'CXPS', 'BIMI', 'AALC', 'AALC', 'ACLN', 'CLDN', 'ABRM', 'AFCO')
# MAGIC )
# MAGIC 
# MAGIC SELECT
# MAGIC   s.ticker, 
# MAGIC   sign(s.slippage), 
# MAGIC   log(-sum(s.slippage)) agg_slippage
# MAGIC FROM slippages s
# MAGIC WHERE sign(s.slippage) = -1
# MAGIC GROUP BY s.ticker, sign(s.slippage)
# MAGIC ORDER BY log(-sum(s.slippage)) DESC
# MAGIC LIMIT 20

# COMMAND ----------

# DBTITLE 1,Show quarterly results of Consistent Price Improvement by Firm
# MAGIC %sql 
# MAGIC 
# MAGIC SELECT 
# MAGIC   week, 
# MAGIC   mpid, 
# MAGIC   price_improv_ct / tot_ct * 100 price_imprv_pct
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     date_trunc('week', s.event_ts) week, 
# MAGIC     s.mpid, 
# MAGIC     count(
# MAGIC       CASE
# MAGIC         WHEN price < asof_ord_time_ask_price AND side_cd = 'B' THEN 1 
# MAGIC         WHEN price > asof_ord_time_bid_price AND side_cd = 'S' THEN 1
# MAGIC       END
# MAGIC     ) price_improv_ct, 
# MAGIC     count(1) tot_ct 
# MAGIC   FROM tempo.silver_trade_slippage s
# MAGIC   GROUP BY date_trunc('week', s.event_ts), s.mpid
# MAGIC ) s

# COMMAND ----------

# DBTITLE 1,Raw Savings Per MPID (Note MPIDs are anonymized) - Higher Percentages Indicate Large Savings Passed on to Customers
# MAGIC %sql 
# MAGIC 
# MAGIC SELECT 
# MAGIC   week, 
# MAGIC   mpid, 
# MAGIC   price_improv_raw
# MAGIC FROM (
# MAGIC     SELECT 
# MAGIC       date_trunc('week', event_ts) week, 
# MAGIC       mpid, 
# MAGIC       count(
# MAGIC         CASE
# MAGIC           WHEN price < asof_ord_time_ask_price AND side_cd = 'B' THEN 1 
# MAGIC           WHEN price > asof_ord_time_bid_price AND side_cd = 'S' THEN 1
# MAGIC         END) price_improv_ct, 
# MAGIC       count(1) tot_ct, 
# MAGIC       sum(
# MAGIC         CASE
# MAGIC           WHEN price < asof_ord_time_ask_price AND side_cd = 'B' THEN size*(asof_ord_time_ask_price - price) 
# MAGIC           WHEN price > asof_ord_time_bid_price AND side_cd = 'S' THEN size*(price - asof_ord_time_bid_price) 
# MAGIC           ELSE 0
# MAGIC         END) price_improv_raw
# MAGIC   FROM tempo.silver_trade_slippage
# MAGIC   WHERE date = '2019-10-01'
# MAGIC   GROUP BY date_trunc('week', event_ts) , mpid
# MAGIC ) s

# COMMAND ----------

# DBTITLE 1,Overall Slippage Over Time
# MAGIC %sql 
# MAGIC 
# MAGIC WITH slippages AS (
# MAGIC   SELECT 
# MAGIC     *,
# MAGIC     date_trunc('week', event_ts) week, 
# MAGIC     CASE 
# MAGIC       WHEN side_cd = 'B' THEN asof_ord_time_ask_price - asof_ex_time_ask_price 
# MAGIC       ELSE asof_ex_time_bid_price - asof_ord_time_bid_price
# MAGIC     END AS slippage 
# MAGIC   FROM tempo.silver_trade_slippage
# MAGIC )
# MAGIC 
# MAGIC SELECT 
# MAGIC   week, 
# MAGIC   sign(slippage), 
# MAGIC   log(-sum(slippage)) agg_slippage
# MAGIC FROM slippages
# MAGIC WHERE sign(slippage) = -1
# MAGIC GROUP BY week, sign(slippage)
# MAGIC ORDER BY week

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### `STEP2` - Summary on price improvement
# MAGIC We've take a few different slides to understand slippage and places where price improvement is offered by firms and by symbol. Running this type of analysis each day and aggregating over the quarter gives retail investors a sense of who offers the best price improvement. Note that a volatility chart will aid in ruling out symbol volatility.

# COMMAND ----------

# DBTITLE 1,Compute Spread as Proxy for Volatility 
daily_volatility = spark.sql("""
select ticker, day, stddev(spread) vol
from (
select ticker, day, spread
from (
      select ticker, date_trunc('day', event_ts) day, asof_ex_time_ask_price - asof_ex_time_bid_price spread 
      from tempo.silver_trade_slippage
      where ticker in ('SWN', 'HAL', 'HQY', 'RGNX', 'BABA', 'LAMR', 'LMT', 'KLAC', 'SMH')
     ) foo
) 
group by ticker, day
order by day
""")
display(daily_volatility)

# COMMAND ----------

# MAGIC %md
# MAGIC ### `STEP3` - Detecting a Basic Spoofing Pattern
# MAGIC 
# MAGIC In order to detect spoofing, we need to understand conditions before order placement, after order placement (was there a full cancellation?) and in between order placement and cancellation. One of the most important aspects is understanding quote changes (often the NBBO with the quotes standing in as a proxy here). We will use the following steps to look for instances of spoofing: 
# MAGIC 
# MAGIC * a) Look for the order placement which pushes the ask price down, followed by an execution on the opposite side, followed by a cancellation of the original order
# MAGIC * b) Look for wash trading
# MAGIC 
# MAGIC If both these are met, we can save the MPID/executions for the date and roll up to alerts at the end of the quarter for further review.
# MAGIC 
# MAGIC <img src = "files/antoine.amend/images/spoofing101.png" width = 750>

# COMMAND ----------

# DBTITLE 1,Mock Up Cancellations - Use ORDER ID to link orders and cancellations in practice
# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS tempo.orders_and_cncls;
# MAGIC 
# MAGIC CREATE TABLE tempo.orders_and_cncls
# MAGIC USING delta 
# MAGIC AS 
# MAGIC SELECT 
# MAGIC   date AS `DATE`, 
# MAGIC   price + 2 price, event_ts + interval 1 seconds AS cncl_event_ts, 
# MAGIC   event_ts - interval 10 seconds AS order_rcvd_ts, 
# MAGIC   10000 order_qt, 
# MAGIC   'Y' AS full_cncl_fl, 
# MAGIC   ticker AS `TICKER`, 
# MAGIC   'S' AS side_cd, 
# MAGIC   mpid
# MAGIC FROM (
# MAGIC   SELECT * FROM tempo.silver_trade_slippage WHERE side_cd = 'B' LIMIT 4000
# MAGIC ) foo1
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   date AS `DATE`, 
# MAGIC   price + 2 price, 
# MAGIC   event_ts + interval 1 seconds AS cncl_event_ts, 
# MAGIC   event_ts - interval 10 seconds AS order_rcvd_ts, 
# MAGIC   10000 order_qt, 
# MAGIC   'N' AS full_cncl_fl, 
# MAGIC   ticker AS `TICKER`, 
# MAGIC   side_cd, 
# MAGIC   mpid
# MAGIC FROM (
# MAGIC   SELECT * FROM tempo.silver_trade_slippage LIMIT 4000
# MAGIC ) foo2;

# COMMAND ----------

# DBTITLE 1,Find Non Bona Fide Orders (with full cancellations) where NBBO moved down from Before Order Receipt Time
from pyspark.sql.functions import * 

prior_quotes = (
  spark.table("tempo.delta_tick_quotes_6mo")
    .filter(col("date") == "2019-07-01")
)

orders_and_cncls = (
  spark.table("tempo.orders_and_cncls")
    .filter(col("full_cncl_fl") == "Y")
    .withColumn("prior_order_rcvd_ts", expr("order_rcvd_ts - interval 10 seconds"))
)


from tempo.tsdf import TSDF
orders_and_cncls_tsdf = TSDF(orders_and_cncls, ts_col = 'prior_order_rcvd_ts', partition_cols = ["DATE", "TICKER"])
prior_quotes_tsdf = TSDF(prior_quotes, ts_col='event_ts', partition_cols = ["DATE", "TICKER"])

prior_order_asof = orders_and_cncls_tsdf.asofJoin(prior_quotes_tsdf, right_prefix = "asof_prior_order")
prior_order_asof = TSDF(prior_order_asof.df, ts_col = 'order_rcvd_ts', partition_cols = ["DATE", "TICKER"])
order_asof = prior_order_asof.asofJoin(prior_quotes_tsdf, right_prefix = "asof_order")

nbbo_deltas = (
  order_asof.df
    .withColumn("nbbo_ask_delta_direction", signum(col("asof_prior_order_ASK_PRICE") - col("asof_order_ASK_PRICE")))
    .withColumn("nbbo_bid_delta_direction", signum(col("asof_order_BID_PRICE") - col("asof_prior_order_BID_PRICE")))
    .withColumn("nbbo_ask_delta", abs(col("asof_prior_order_ASK_PRICE") - col("asof_order_ASK_PRICE")))
    .withColumn("nbbo_bid_delta", abs(col("asof_order_BID_PRICE") - col("asof_prior_order_BID_PRICE")))
)

nbbo_deltas.write.mode('overwrite').format("delta").saveAsTable("tempo.silver_order_nbbo_change")

# COMMAND ----------

# DBTITLE 1,Find all Sell-Side Orders where NBBO Changed Direction Downward
# MAGIC %sql 
# MAGIC SELECT 'order as of time' ts_type, order_rcvd_ts, asof_prior_order_event_ts, asof_order_event_ts, nbbo_ask_delta_direction, side_cd, asof_prior_order_ask_price ts_price 
# MAGIC FROM tempo.silver_order_nbbo_change
# MAGIC WHERE nbbo_ask_delta_direction > 0
# MAGIC AND side_cd = 'S'
# MAGIC AND order_rcvd_ts BETWEEN '2019-07-01 14:00:00' AND '2019-07-01 14:30:00'
# MAGIC UNION ALL
# MAGIC SELECT 'order shifted backward as of time' ts_type, order_rcvd_ts, asof_prior_order_event_ts, asof_order_event_ts, nbbo_ask_delta_direction, side_cd, asof_order_ask_price ts_price
# MAGIC FROM tempo.silver_order_nbbo_change
# MAGIC WHERE nbbo_ask_delta_direction > 0
# MAGIC AND side_cd = 'S'
# MAGIC AND order_rcvd_ts BETWEEN '2019-07-01 14:00:00' AND '2019-07-01 14:30:00'

# COMMAND ----------

# DBTITLE 1,Overlay Quote Changes with Order Executions on Opposite Side
# MAGIC %sql 
# MAGIC 
# MAGIC DROP TABLE IF EXISTS tempo.gold_non_bona_fide_executions;
# MAGIC 
# MAGIC CREATE TABLE tempo.gold_non_bona_fide_executions
# MAGIC USING delta AS
# MAGIC SELECT 
# MAGIC   event_ts AS ex_event_ts, 
# MAGIC   x.date, 
# MAGIC   x.ticker, 
# MAGIC   x.price, 
# MAGIC   size, 
# MAGIC   x.mpid, 
# MAGIC   marketParticipantName
# MAGIC FROM tempo.silver_trade_slippage x
# MAGIC JOIN tempo.silver_order_nbbo_change o
# MAGIC ON x.mpid = o.mpid
# MAGIC AND x.ticker = o.ticker
# MAGIC AND x.date = o.date
# MAGIC WHERE (x.side_cd = 'B' AND o.side_cd = 'S')
# MAGIC AND nbbo_ask_delta_direction > 0
# MAGIC AND x.event_ts BETWEEN o.order_rcvd_ts AND o.cncl_event_ts;
# MAGIC 
# MAGIC SELECT * FROM tempo.gold_non_bona_fide_executions

# COMMAND ----------

# DBTITLE 1,Identify Wash Trading (Activity of MPID w/ itself)
# MAGIC %sql 
# MAGIC 
# MAGIC DROP TABLE IF EXISTS tempo.gold_wash_trades;
# MAGIC 
# MAGIC CREATE TABLE tempo.gold_wash_trades 
# MAGIC AS 
# MAGIC SELECT 
# MAGIC   mpid, 
# MAGIC   event_ts, 
# MAGIC   size, 
# MAGIC   price, 
# MAGIC   ticker, 
# MAGIC   sum(side_nb) sum_side_nb
# MAGIC FROM (
# MAGIC   SELECT mpid, event_ts, size, price, ticker, 1 side_nb
# MAGIC   FROM tempo.silver_trade_slippage 
# MAGIC   WHERE date = '2019-07-01'
# MAGIC   AND side_cd = 'B'
# MAGIC   UNION ALL
# MAGIC   SELECT mpid, event_ts, size, price, ticker, -1 side_nb
# MAGIC   FROM tempo.silver_trade_slippage 
# MAGIC   WHERE date = '2019-07-01'
# MAGIC   AND side_cd = 'S'
# MAGIC ) foo
# MAGIC GROUP BY mpid, event_ts, size, price, ticker
# MAGIC HAVING sum(side_nb) = 0;
# MAGIC 
# MAGIC SELECT * FROM tempo.gold_wash_trades

# COMMAND ----------

# DBTITLE 1,Save Off Report of Overlap of Non Bona-fide Executions & Wash Trades as Spoofing Alert
# MAGIC %sql 
# MAGIC 
# MAGIC SELECT 
# MAGIC   a.mpid, 
# MAGIC   min(a.size) min_size, 
# MAGIC   max(a.size) max_size, 
# MAGIC   log(count(1)) log_trade_ct
# MAGIC FROM tempo.gold_non_bona_fide_executions a 
# MAGIC JOIN tempo.gold_wash_trades b 
# MAGIC ON a.mpid = b.mpid 
# MAGIC AND a.size = b.size
# MAGIC GROUP BY a.mpid
# MAGIC ORDER BY log_trade_ct DESC
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC In this series of simple analytics powered by a robust time series processing engine, we've run a few examples of market surveillance which pertain to the equities and digital currency spaces. 
# MAGIC Databricks helps in all use cases of market surveillance by providing:
# MAGIC 
# MAGIC * Reduce TCO on ETL performance using Databricks runtime
# MAGIC * `tempo` library to speed up development time and best practices by modularizing common code
# MAGIC * Faster time-to-market with combined SQL + Python + Scala environment

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC + <a href="$./01_tempo_context">STAGE0</a>: Home page
# MAGIC + <a href="$./02_tempo_etl">STAGE1</a>: Design pattern for ingesting BPipe data
# MAGIC + <a href="$./03_tempo_volatility">STAGE2</a>: Introducing tempo for calculating market volatility
# MAGIC + <a href="$./04_tempo_spoofing">STAGE3</a>: Introducing tempo for market surveillance
# MAGIC ---
