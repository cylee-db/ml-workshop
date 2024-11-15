# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory data analysis

# COMMAND ----------

# MAGIC %md
# MAGIC # Query and profile data
# MAGIC
# MAGIC Query data from your catalogs.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM samples.nyctaxi.trips
# MAGIC ;

# COMMAND ----------

# MAGIC %md
# MAGIC # Work with SQL query results
# MAGIC
# MAGIC The `_sqldf` variable holds a Spark Dataframe of the SQL query result.

# COMMAND ----------

taxi_df = _sqldf

# COMMAND ----------

display(taxi_df.select("fare_amount"))

# COMMAND ----------

# MAGIC %md
# MAGIC Convert the Spark Dataframe to a pandas dataframe if it fits into the memory of your VM.

# COMMAND ----------

taxi_pdf = taxi_df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # Work with plotting libraries

# COMMAND ----------

import seaborn as sns

g = sns.PairGrid(taxi_pdf.sample(frac=0.1), diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3)
g.map_upper(sns.regplot)

# COMMAND ----------

# MAGIC %md
# MAGIC # Specify your working catalog and database
# MAGIC
# MAGIC Widgets are a convenient way of providing variables to your notebook.
# MAGIC
# MAGIC Run the following cell to create two widgets.

# COMMAND ----------

dbutils.widgets.text("catalog", "<catalog_name>")
dbutils.widgets.text("db", "<db_name>")

# COMMAND ----------

# MAGIC %md
# MAGIC Enter your working catalog and database names in the widget. Run the next cell to capture these values. We will be saving a table under them.

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE DATABASE {db}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create a new table
# MAGIC
# MAGIC Run the next cell to create a new table in your working database.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS taxi_trips
# MAGIC AS SELECT * FROM samples.nyctaxi.trips
# MAGIC ;
