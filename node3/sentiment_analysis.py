import mlflow
import os
import infinstor_mlflow_plugin
import boto3
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import tempfile
import pandas as pd
import pickle
import json
import sys
from concurrent_plugin import concurrent_core
from transformers import pipeline

print('sentiment_analysis: Entered', flush=True)
df = concurrent_core.list(None)
if df.isempty:
    print('No input files. Done', flush=True)
    os._exit(os.EX_OK)

lp = concurrent_core.get_local_paths(df)
consolidated_pd = pd.DataFrame()
for one_local_path in lp:
    print('Begin processing file ' + str(one_local_path), flush=True)
    try:
        df1 = pd.read_pickle(one_local_path)
    except Exception is ex:
        print('Error ' + str(ex) + ' processing file ' + str(one_local_path), flush=True)
        continue
    if consolidated_pd.empty:
        consolidated_pd = df1
    else:
        consolidated_pd = pd.DataFrame(df1)
consolidated_pd.reset_index()

positives = 0
negatives = 0
positives_samples = []
negatives_samples = []
def do_count(row):
    global negatives
    global negatives_samples
    global positives
    global positives_samples
    print("do_count: Entered. sentiment=" + row['label'] + ", score=" + str(row['score']) + ", ner=" + str(row['ner']))
    if row['label'] == 'NEGATIVE' and row['score'] > 0.9:
        negatives = negatives + 1
        if negatives < 1000:
            negatives_samples.append({'screen_name': row['screen_name'], 'text': row['text']})
    if row['label'] == 'POSITIVE' and row['score'] > 0.9:
        positives = positives + 1
        if positives < 1000:
            positives_samples.append({'screen_name': row['screen_name'], 'text': row['text']})
    return None

consolidated_pd.apply(do_count, axis=1)

fn = "/tmp/positives_samples.json"
with open(fn, 'w') as f:
    json.dump(positives_samples, f, ensure_ascii=True, indent=4)
concurrent_core.concurrent_log_artifact(fn, "")

fn = "/tmp/negatives_samples.json"
with open(fn, 'w') as f:
    json.dump(negatives_samples, f, ensure_ascii=True, indent=4)
concurrent_core.concurrent_log_artifact(fn, "")

print('positives=' + str(positives) + ', negatives=' + str(negatives), flush=True)
fn = "/tmp/sentiment_summary.json"
with open(fn, 'w') as f:
    f.write(json.dumps({'positives': positives, 'negatives': negatives}))
concurrent_core.concurrent_log_artifact(fn, "")

os._exit(os.EX_OK)
