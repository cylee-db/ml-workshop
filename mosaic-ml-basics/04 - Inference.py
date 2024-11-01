# Databricks notebook source
# MAGIC %md
# MAGIC # Inference
# MAGIC
# MAGIC At this point, you would have had a model registered to Unity Catalog. If not, go to the MLflow experiment from the previous step, choose a model and register it to Unity Catalog, under your working catalog and database. Give the model version an alias, e.g. `@Challenger` or `@dev`.
# MAGIC
# MAGIC We will use this model for inference.

# COMMAND ----------

# MAGIC %md
# MAGIC # Specify your working catalog, database and model name
# MAGIC
# MAGIC Widgets are a convenient way of providing variables to your notebook.
# MAGIC
# MAGIC Run the following cell to create the widgets.

# COMMAND ----------

dbutils.widgets.text("catalog", "<catalog_name>")
dbutils.widgets.text("db", "<db_name>")
dbutils.widgets.text("table_name", "taxi_trips")
dbutils.widgets.text("model_name", "taxi_fare")
dbutils.widgets.text("model_alias", "@Champion")

# COMMAND ----------

# MAGIC %md
# MAGIC Enter your working catalog, database, model names and alias in the widget. Run the next cell to capture these values.
# MAGIC
# MAGIC Be sure to use the same model name and alias as the model you registered to Unity Catalog.

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")
table_name = dbutils.widgets.get("table_name")
model_name = dbutils.widgets.get("model_name")
model_alias = dbutils.widgets.get("model_alias")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE DATABASE {db}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Read the data
# MAGIC
# MAGIC Note how we reference the parameter `:table_name` in a SQL query.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM
# MAGIC   identifier(:table_name)
# MAGIC ;

# COMMAND ----------

# MAGIC %md
# MAGIC We will use only the `trip_distance` column to generate predictions.

# COMMAND ----------

df = _sqldf.select('trip_distance')
pdf = df.toPandas()

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load the model
# MAGIC
# MAGIC To perform inference in batch mode against a dataset, there are two options:
# MAGIC
# MAGIC - Predict on records in a pandas dataframe if the amount of data fits onto your VM.
# MAGIC - Predict on records in a Spark dataframe when there is a huge amount of data. You will need a cluster to better leverage distributed processing.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict on a pandas dataframe
# MAGIC
# MAGIC We will load the model into a `pyfunc` model.

# COMMAND ----------

import mlflow

model_uri = f'models:/{catalog}.{db}.{model_name}{model_alias}'

print(f'Loading model: {model_uri}')

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Predict on a Pandas DataFrame.
predictions = loaded_model.predict(pdf)

predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict on a Spark dataframe
# MAGIC
# MAGIC We will load the model as a Spark UDF.

# COMMAND ----------

from pyspark.sql.functions import struct, col

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_spark_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# Predict on a Spark DataFrame.
predictions_df = df.withColumn(
    "predictions",
    loaded_spark_model(*loaded_spark_model.metadata.get_input_schema().input_names()),
)

display(predictions_df)
