# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare feature table and function
# MAGIC
# MAGIC We will prepare the feature table and feature function for training our model.
# MAGIC
# MAGIC The feature table will also be where features are retrived during inference time.

# COMMAND ----------

# MAGIC %md
# MAGIC # Specify your working catalog and database
# MAGIC
# MAGIC Run the following cell to create two widgets. Enter the names of your working catalog and database.

# COMMAND ----------

dbutils.widgets.text("catalog", "<your_working_catalog_name>")
dbutils.widgets.text("db", "<your_working_db_name>")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE DATABASE {db}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data exploration and analysis
# MAGIC
# MAGIC Let's review our dataset and start analyze the data we have to predict our churn

# COMMAND ----------

# Read our churn_features table
churn_dataset = spark.table("ml_churn_features")
display(churn_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data analysis and preparation using Pandas On Spark API
# MAGIC
# MAGIC If you are familiar with Pandas, you can easily use the [pandas on spark API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/index.html) to scale `pandas` code on Spark to process large scale data. The Pandas instructions will be converted in the spark engine under the hood and distributed at scale.
# MAGIC
# MAGIC Use `pandas_api()` to get a Pandas-on-Spark Dataframe from a Spark Dataframe.
# MAGIC
# MAGIC Continue to use the familiar Pandas syntax without having to learn Spark while you process large datasets.

# COMMAND ----------

# Convert to pandas-on-Spark
dataset = churn_dataset.pandas_api()
dataset.describe()
# Drop columns we don't want to use in our model
dataset = dataset.drop(
    columns=[
        "address",
        "email",
        "firstname",
        "lastname",
        "last_activity_date",
        "last_event",
    ]
)
# Drop missing values
dataset = dataset.dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC # Write features dataset to the Feature Store
# MAGIC
# MAGIC We will write the features dataset as a table called `churn_user_features` in your working database.
# MAGIC
# MAGIC In Databricks, __any table with a primary key can be used as a feature table__. This is the so-called Feature Store of Databricks.
# MAGIC
# MAGIC After writing the table, we will go on to set a primary key on the table.

# COMMAND ----------

features_df = dataset.to_spark()

features_df.write.mode("overwrite").saveAsTable(f"{catalog}.{db}.churn_user_features")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM churn_user_features
# MAGIC ;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Primary key column cannot be null
# MAGIC ALTER TABLE churn_user_features ALTER COLUMN user_id SET NOT NULL
# MAGIC ;
# MAGIC -- Set the primary key column
# MAGIC ALTER TABLE churn_user_features ADD CONSTRAINT churn_user_features_pk PRIMARY KEY(user_id)
# MAGIC ;

# COMMAND ----------

# MAGIC %md
# MAGIC That's it. Our feature table is ready for use.

# COMMAND ----------

# MAGIC %md
# MAGIC # Calculate user tenure feature using a function
# MAGIC
# MAGIC There may be some features that we don't want to keep in the feature table. For example, a __user's tenure in days__ is a feature that keeps changing everyday, and we don't want refresh our feature pipeline all the time to calculate a new value daily.
# MAGIC
# MAGIC To do this, we can define a function that calculates tenure only when it's needed, for example, when performing inference.
# MAGIC
# MAGIC Functions are registered to Unity Catalog.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION user_tenure(creation_date_in TIMESTAMP)
# MAGIC RETURNS INT
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT "[Feature Function] Calculate user tenure in days based on their creation date"
# MAGIC AS $$
# MAGIC from datetime import datetime
# MAGIC if creation_date_in is not None:
# MAGIC   diff = datetime.now() - creation_date_in.replace(tzinfo=None)
# MAGIC   days = diff.days
# MAGIC else:
# MAGIC   days = -1
# MAGIC return days
# MAGIC $$;
# MAGIC ;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Test the function
# MAGIC SELECT user_tenure('2024-11-01') as nb_days
# MAGIC ;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT user_tenure(creation_date) as user_tenure
# MAGIC FROM churn_user_features
# MAGIC ORDER by user_tenure
# MAGIC ;
