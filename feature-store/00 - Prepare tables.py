# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare tables
# MAGIC
# MAGIC We will be preparing the tables for this workshop from a demo that has been installed in your workspace.
# MAGIC
# MAGIC We will prepare two tables - one for the potential features, and one with the training labels.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Specify the demo catalog and database
# MAGIC
# MAGIC Run the following cell to create two widgets. Enter the names of the catalog and database where the demo is installed.

# COMMAND ----------

dbutils.widgets.text("demo_catalog", "databricks_demos")
dbutils.widgets.text("demo_db", "lakehouse_c360")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Specify your working catalog and database
# MAGIC
# MAGIC Run the following cell to create two more widgets. Now, enter the names of your working catalog and database.

# COMMAND ----------

dbutils.widgets.text("catalog", "<your_working_catalog_name>")
dbutils.widgets.text("db", "<your_working_db_name>")

# COMMAND ----------

# MAGIC %md
# MAGIC Enter your working catalog and database names in the widget. Run the next cell to capture these values. We will be saving a table under them.

# COMMAND ----------

demo_catalog = dbutils.widgets.get("demo_catalog")
demo_db = dbutils.widgets.get("demo_db")
catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE DATABASE {db}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the two tables

# COMMAND ----------

import pandas as pd

pdf = pd.read_csv(f"./data/churn_features.csv",
                  parse_dates=[2, 3, 15, 19])

# Alternatively, upload csv to a Volume in your working schema and read from there
# pdf = pd.read_csv(f"/Volumes/{catalog}/{db}/wkshp_volume/churn_features.csv",
#                   parse_dates=[2, 3, 15, 19])

df = spark.createDataFrame(pdf)

display(df)

# COMMAND ----------

features_df = df.drop("churn")

features_df.write.mode("overwrite").saveAsTable(f"{catalog}.{db}.ml_churn_features")

# COMMAND ----------

from pyspark.sql.functions import col

labels_df = df.select("user_id", col('churn').alias("churn_label"))

labels_df.write.mode("overwrite").saveAsTable(f"{catalog}.{db}.ml_churn_labels")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Primary key column cannot be null
# MAGIC ALTER TABLE ml_churn_labels ALTER COLUMN user_id SET NOT NULL
# MAGIC ;
# MAGIC -- Set the primary key column
# MAGIC ALTER TABLE ml_churn_labels ADD CONSTRAINT user_pk PRIMARY KEY(user_id)
# MAGIC ;

# COMMAND ----------

# MAGIC %md
# MAGIC ## ALTERNATIVE: Set up from the Lakehouse C360 demo

# COMMAND ----------

# features_df = spark.read.table(f"{demo_catalog}.{demo_db}.churn_features").drop("churn")

# features_df.write.mode("overwrite").saveAsTable(f"{catalog}.{db}.ml_churn_features")

# COMMAND ----------

# %sql
# DROP TABLE IF EXISTS ml_churn_labels
# ;
# CREATE TABLE IF NOT EXISTS ml_churn_labels (
#   user_id STRING,
#   churn_label INT,
#   CONSTRAINT user_pk PRIMARY KEY (user_id)
# )
# ;

# COMMAND ----------

# query = f"""
#   MERGE INTO {catalog}.{db}.ml_churn_labels t USING {demo_catalog}.{demo_db}.churn_features s
#   ON t.user_id = s.user_id
#   WHEN NOT MATCHED THEN
#   INSERT
#     (user_id, churn_label)
#   VALUES(
#       s.user_id,
#       s.churn
#     )
#   ;
#   """
  
# spark.sql(query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect the tables
# MAGIC
# MAGIC The two tables are now set up for the exercise.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ml_churn_features
# MAGIC ;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ml_churn_labels
# MAGIC ;
