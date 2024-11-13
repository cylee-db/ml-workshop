# Databricks notebook source
# MAGIC %md
# MAGIC # Batch inference
# MAGIC
# MAGIC We will perform batch inference using the deployed model against a list of user IDs.

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
# MAGIC # Data for scoring
# MAGIC
# MAGIC We will create a new table with only the `user_id` column to perform inference.
# MAGIC
# MAGIC Since the model is logged with the feature information, it knows to go to the Feature Store to retrieve the features for each user when given a table with only the user ID. Remember that any table that has a primary key in Unity Catalog can be used as a feature table.

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS ml_user_ids
# MAGIC ;
# MAGIC CREATE TABLE IF NOT EXISTS ml_user_ids (
# MAGIC   user_id STRING,
# MAGIC   CONSTRAINT user_id_pk PRIMARY KEY (user_id)
# MAGIC )
# MAGIC ;
# MAGIC INSERT INTO ml_user_ids
# MAGIC SELECT user_id FROM churn_user_features
# MAGIC ;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ml_user_ids
# MAGIC ;

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import mlflow

# Batch inference

# If the model at model_uri is packaged with the features, the FeatureStoreClient.score_batch()
# call automatically retrieves the required features from Feature Store before scoring the model.
# The DataFrame returned by score_batch() augments batch_df with
# columns containing the feature values and a column containing model predictions.

model_name="fs_customer_churn"

fe = FeatureEngineeringClient()

batch_df = fe.read_table(name=f'{catalog}.{db}.ml_user_ids')

# batch_df only has the user_id column
predictions_df = fe.score_batch(
    model_uri=f"models:/{catalog}.{db}.{model_name}@challenger",
    df=batch_df
)

display(predictions_df)
