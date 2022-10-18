import mlflow
import os
import infinstor_mlflow_plugin
import boto3
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import pandas as pd
import pickle
import json
import sys
from concurrent_plugin import concurrent_core
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

print('sentiment_analysis: Entered', flush=True)
df = concurrent_core.list(None, input_name='tweets')

print('Column Names:', flush=True)
cn = df.columns.values.tolist()
print(str(cn))

print('------------------------------ Obtained input info ----------------', flush=True)

for ind, row in df.iterrows():
    print("Input row=" + str(row), flush=True)

print('------------------------------ Finished dump of input info ----------------', flush=True)

lp = concurrent_core.get_local_paths(df)

print('Location paths=' + str(lp))

print('------------------------------ Begin Loading Huggingface sentiment-analysis Pipeline ------------------', flush=True)
nlp = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
print('------------------------------ After Loading Huggingface sentiment-analysis Pipeline ------------------', flush=True)

def do_nlp_fnx(row):
    s = nlp(row['text'])[0]
    return [s['label'], s['score'], row['user']['screen_name']]

print('------------------------------ Before Inference ------------------', flush=True)
consolidated_pd = pd.DataFrame()
for one_local_path in lp:
    print('Begin processing file ' + str(one_local_path), flush=True)
    jsonarray = pickle.load(open(one_local_path, 'rb'))
    # for i in jsonarray:
    #   print(json.dumps(i), flush=True)
    df1 = pd.DataFrame(jsonarray, columns=['text', 'user'])
    if consolidated_pd.empty:
        consolidated_pd = df1
    else:
        consolidated_pd = pd.DataFrame(df1)

consolidated_pd[['label', 'score', 'screen_name']] = consolidated_pd.apply(do_nlp_fnx, axis=1, result_type='expand')
consolidated_pd.reset_index()

tfname = "/tmp/output.pickle"
if os.path.exists(tfname):
    os.remove(tfname)
consolidated_pd.to_pickle(tfname)
concurrent_core.concurrent_log_artifact(tfname, "")
print('Finished logging artifacts file')
print('------------------------------ After Inference. End ------------------', flush=True)

os._exit(os.EX_OK)
