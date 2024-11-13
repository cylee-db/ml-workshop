# Databricks notebook source
# MAGIC %md
# MAGIC # Train and register model
# MAGIC
# MAGIC We are now ready to train the model.
# MAGIC
# MAGIC We will retrieve features from the Feature Store for training. We will also let the feature store compute the user tenure feature on the fly.
# MAGIC
# MAGIC After training the model, we will register it to Unity Catalog.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Specify your working catalog and database
# MAGIC
# MAGIC Run the following cell to create two more widgets. Now, enter the names of your working catalog and database.

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
# MAGIC
# MAGIC Recall that these are the object we created before:
# MAGIC
# MAGIC - `churn_user_features` - Feature table with cleaned features
# MAGIC - `ml_churn_labels` - Label table
# MAGIC - `user_tenure` - Function to calculate tenure feature

# COMMAND ----------

# MAGIC  %md
# MAGIC # Define features and prepare training data
# MAGIC
# MAGIC  The `databricks.feature_engineering` library lets us define feature lookups and feature functions to assemble the training dataset.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup, FeatureFunction

# The model training uses all features from the feature table
feature_lookups = [
    FeatureLookup(
      table_name=f'{catalog}.{db}.churn_user_features',
      feature_names=None,   # Looks up all features in the table
      lookup_key='user_id'
    ),
    FeatureFunction(
      udf_name=f"{catalog}.{db}.user_tenure",
      input_bindings={
        "creation_date_in" : "creation_date"
      },
      output_name="user_tenure"
    )
  ]

fe = FeatureEngineeringClient()

# Use fe.read_table() to read the table with the pk and label column
# This is used as the LHS table to join with the feature table
training_df = fe.read_table(name=f'{catalog}.{db}.ml_churn_labels')

# When we call fe.create_training_set(), it uses the PK in the LHS table to lookup
# features from the tables specified in feature_lookup.

# Create a training set using training DataFrame and features from Feature Store
# The training DataFrame must contain all lookup keys from the set of feature lookups.
# It must also contain all labels used for training, in this case the 'churn_label' column.
training_set = fe.create_training_set(
  df=training_df,
  feature_lookups=feature_lookups,
  label='churn_label',
  exclude_columns=['user_id', 'last_transaction', 'creation_date']
)

# Inspect the assembled training data
display(training_set.load_df())

# We have constructed the training set.
# However, fe.create_training_set() has to be called from within mlflow.start_run()
# in order for the data-model lineage to be captured.
# So we'll repeat this in the next cell.

# COMMAND ----------

import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import lightgbm
from lightgbm import LGBMClassifier
import mlflow
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

def evaluate(model, X, y, prefix):
  return mlflow.evaluate(
      model,
      X.join(y),
      targets="churn_label",
      model_type="classifier",
      evaluators="default",
      evaluator_config={"metric_prefix": prefix, "pos_label": 1},
   ).metrics


with mlflow.start_run() as mlflow_run:

  # Create a training set using training DataFrame and features from Feature Store
  # The training DataFrame must contain all lookup keys from the set of feature lookups,
  # in this case 'customer_id' and 'product_id'. It must also contain all labels used
  # for training, in this case 'rating'.
  training_set = fe.create_training_set(
    df=training_df,
    feature_lookups=feature_lookups,
    label='churn_label',
    exclude_columns=['user_id', 'last_transaction', 'creation_date']
  )

  training_pdf = training_set.load_df().toPandas()

  X_train, X_test, y_train, y_test = train_test_split(
    training_pdf.drop(['churn_label'], axis=1),
    training_pdf['churn_label'],
    test_size=0.4, random_state=42
  )

  X_val, X_test, y_val, y_test = train_test_split(
      X_test, y_test, test_size=0.5, random_state=42
  )

  tx_pipeline = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])
  tx_pipeline.fit(X_train, y_train)

  X_val_tx = tx_pipeline.transform(X_val)

  params = {
    'max_depth': -1,
    'n_estimators': 100,
    'learning_rate': 0.1
  }

  lgbmc_classifier = LGBMClassifier(**params)

  model = Pipeline([('tx', tx_pipeline),
                    ('lgbmc', lgbmc_classifier)]
                  )

  model = model.fit(X_train, y_train,
      lgbmc__callbacks=[lightgbm.early_stopping(5), lightgbm.log_evaluation(0)],
      lgbmc__eval_set=[(X_val_tx, y_val)])

  # Log the model using the feature engineering client.
  # This will register the feature lineage.
  fe.log_model(
    model=model,
    artifact_path="model",
    flavor=mlflow.lightgbm,
    training_set=training_set
  )

  # The model logged with the FE client is not compatible with mlflow.evaluate
  # So we have log a sklearn flavor of it and retrieve it to use mlflow.evaluate
  mlflow.sklearn.log_model(model,
    artifact_path='sklearn_model',
    signature=infer_signature(X_train, y_train))

  # We also have to manually log the model params since the FE client doesn't autolog
  mlflow.log_params(model.get_params())

  # Log the dataset used in this run
  # This will reference the UC table from the Experiment UI
  dataset = mlflow.data.load_delta(table_name=f"{catalog}.{db}.churn_user_features")
  mlflow.log_input(dataset, context="train-val-test")

  # Now, we load the sklearn flavor that we just logged
  loaded_model = mlflow.pyfunc.load_model(f"{mlflow_run.info.artifact_uri}/sklearn_model")
    
  # evaluate() is a wrapper function here
  # It calls mlflow.evaluate(), which autologs the evluation metrics
  
  # Log metrics for the training set
  lgbmc_training_metrics = evaluate(loaded_model, X_train, y_train, "training_")

  # Log metrics for the validation set
  lgbmc_val_metrics = evaluate(loaded_model, X_val, y_val, "val_")

  # Log metrics for the test set
  lgbmc_test_metrics = evaluate(loaded_model, X_test, y_test, "test_")

  # Log the evaluation data
  eval_data = X_test
  eval_data["churn_label"] = y_test
  eval_data["predictions"] = loaded_model.predict(X_test)
  mlflow.log_table(eval_data, "evaluation/test_data.json")

# COMMAND ----------

# MAGIC %md
# MAGIC # Register model to Unity Catalog

# COMMAND ----------

# ID of last run
run_id = mlflow_run.info.run_id

model_name = "fs_customer_churn"

registered_model = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=f"{catalog}.{db}.{model_name}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Add a `@challenger` alias to the latest model version.

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

client.set_registered_model_alias(
    name=f"{catalog}.{db}.{model_name}",
    version=registered_model.version,
    alias="challenger",
)
