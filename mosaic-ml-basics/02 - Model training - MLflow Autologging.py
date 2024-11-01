# Databricks notebook source
# MAGIC %md
# MAGIC # Model training
# MAGIC
# MAGIC We will train a model using the NYC taxi dataset to predict the fare amount.
# MAGIC
# MAGIC We will use this example to look at how you can track the training runs in an MLflow experiment. We will use MLflow's Autologging to see what it logs for us.

# COMMAND ----------

# MAGIC %md
# MAGIC # Specify your working catalog, database and table
# MAGIC
# MAGIC Widgets are a convenient way of providing variables to your notebook.
# MAGIC
# MAGIC Run the following cell to create three widgets.

# COMMAND ----------

dbutils.widgets.text("catalog", "<catalog_name>")
dbutils.widgets.text("db", "<db_name>")
dbutils.widgets.text("table_name", "taxi_trips")

# COMMAND ----------

# MAGIC %md
# MAGIC Enter your working catalog, database and table names in the widget. Run the next cell to capture these values.
# MAGIC
# MAGIC If you did not create a table from the previous step, use:
# MAGIC
# MAGIC - catalog = `samples`
# MAGIC - db = `nyctaxi`
# MAGIC - table_name = `trips`

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")

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
# MAGIC We will use only the `trip_distance` and `fare_amount` columns.

# COMMAND ----------

df = _sqldf.select('trip_distance', 'fare_amount').toPandas()
df

# COMMAND ----------

# MAGIC %md
# MAGIC # Perform train/test split
# MAGIC
# MAGIC Note: You may get an error running the next cell for the first time. This is due to the import statement. Run the cell for a second time and the error will be gone.

# COMMAND ----------

from sklearn.model_selection import train_test_split

label_df = df[['fare_amount']].values.ravel()
features_df = df.drop(['fare_amount'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(features_df, label_df, test_size=0.2)

# COMMAND ----------

print(f"Train and test size: {len(x_train)}, {len(x_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Train (with autologging)
# MAGIC
# MAGIC Run the following cell and check the MLflow experiments pane to see what MLflow autologs for you. Note that we haven't had to write any line of code involving MLflow yet.

# COMMAND ----------

# With autologging

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC # Train (without autologging)
# MAGIC
# MAGIC We will now turn off MLflow Autologging and see what happens when we run `fit()`.

# COMMAND ----------

# Without autologging

import mlflow

mlflow.sklearn.autolog(disable=True)

lr.fit(x_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC # Restore MLflow Autologging
# MAGIC
# MAGIC Enable MLflow Autologging again.

# COMMAND ----------

mlflow.sklearn.autolog(disable=False)
