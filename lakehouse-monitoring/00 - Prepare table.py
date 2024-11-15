# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare table
# MAGIC
# MAGIC We will be preparing a table that we will monitor.

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

catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE DATABASE {db}")

# COMMAND ----------

import pandas as pd
import random

date_time = pd.date_range(start='2024-11-01', end='2024-11-04', freq='H').to_list()[:-1]
location = random.choices(['NY', 'LA', 'SF'], k=72)
temperature = random.choices(range(70, 80), k=72)

pdf1 = pd.DataFrame({'date': date_time, 'location': location, 'temperature': temperature})

date_time = pd.date_range(start='2024-11-04', end='2024-11-07', freq='H').to_list()[:-1]
location = random.choices(['NY', 'LA', None], k=72)
temp = random.choices(range(70, 80), k=72)
temperature = [x if random.uniform(0, 1) > 0.2 else None for x in temp]

pdf2 = pd.DataFrame({'date': date_time, 'location': location, 'temperature': temperature})

date_time = pd.date_range(start='2024-11-07', end='2024-11-10', freq='H').to_list()[:-1]
location = random.choices(['NY', 'LA', 'SF', 'Chicago'], k=72)
temperature = random.choices(range(70, 80), k=36) + random.choices(range(40, 50), k=36)

pdf3 = pd.DataFrame({'date': date_time, 'location': location, 'temperature': temperature})

pdf = pd.concat([pdf1, pdf2, pdf3], ignore_index=True)

spark.createDataFrame(pdf).write.mode("overwrite").saveAsTable(f"{catalog}.{db}.temperature_drift")

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE temperature_drift SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM temperature_drift
# MAGIC ;
