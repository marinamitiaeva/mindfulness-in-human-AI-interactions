import os
import json
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, FloatType
import wordsegment

# Initialize wordsegment
wordsegment.load()

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MoralFormerScoring") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Load data
input_path = "data/politeness_results_normalized_shrt.json"
df = pd.read_json(input_path)
df['internal_id'] = range(len(df))

# Split into chunks (example: 6 chunks)
num_chunks = 6
chunk_size = len(df) // num_chunks
dfs = [df.iloc[i*chunk_size:(i+1)*chunk_size] if i < num_chunks - 1 else df.iloc[i*chunk_size:] for i in range(num_chunks)]

# Pick a chunk to process (example: last chunk)
df_spark = spark.createDataFrame(dfs[-1])

# Define moral foundations and load model
FOUNDATIONS = ["care", "fairness", "loyalty", "authority", "sanctity"]
model_name = "microsoft/moralformer-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# UDF to compute moral scores
def get_morality_scores(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().tolist()
        return {FOUNDATIONS[i]: float(probs[i]) for i in range(len(FOUNDATIONS))}
    except Exception as e:
        return {foundation: None for foundation in FOUNDATIONS}

# Register UDF
morality_udf = udf(lambda text: json.dumps(get_morality_scores(text)), StringType())

# Apply UDF to Spark DataFrame
result_df = df_spark.withColumn("morality_scores", morality_udf(col("value")))

# Convert back to Pandas
result_pd = result_df.toPandas()

# Expand JSON scores into columns
morality_expanded = result_pd["morality_scores"].apply(json.loads).apply(pd.Series)
result_final = pd.concat([result_pd.drop(columns=["morality_scores"]), morality_expanded], axis=1)

# Save the result
output_path = "data/processed/morality_scores.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result_final.to_json(output_path, orient="records", indent=2, force_ascii=False)

print(f"Saved moral scores for {len(result_final)} entries to {output_path}")