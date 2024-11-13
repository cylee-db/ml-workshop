# Databricks notebook source
# MAGIC %md
# MAGIC # Online inference

# COMMAND ----------

# MAGIC %md
# MAGIC To create a REST API endpoint that can be called for online inference, do the following:
# MAGIC
# MAGIC - Create an Online Table from the feature table `churn_user_features`
# MAGIC - Create a Serving Endpoint for the registered model `fs_customer_churn`
# MAGIC - Test the endpoint using the following json input

# COMMAND ----------

model_input = """
{
    "inputs": [
        {"user_id": "dddede45-aa29-4a2a-abb0-e6b5259a72b2"},
        {"user_id": "697fe5f3-022a-48c1-b65f-f79d8ea6ed1c"},
        {"user_id": "4f4451df-8a6f-452e-ad05-dcb5559f8dd4"},
        {"user_id": "4734d799-1260-4d2e-b075-e8ada56fa902"}
    ]
}
"""

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively, use the following code to invoke the REST API endpoint.
# MAGIC
# MAGIC Change the model endpoint name to the name of your endpoint.

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

model_endpoint_name = "fs_customer_churn_cyl"

def score_model(dataset):
  url = f'https://{dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()}/serving-endpoints/{model_endpoint_name}/invocations'
  headers = {'Authorization': f'Bearer {dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()}', 'Content-Type': 'application/json'}

  data_json = dataset
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

score_model(model_input)
