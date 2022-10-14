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
df = concurrent_core.list(None)

print('Column Names:', flush=True)
cn = df.columns.values.tolist()
print(str(cn))

print('------------------------------ Obtained input info ----------------', flush=True)

for ind, row in df.iterrows():
    print("Input row=" + str(row), flush=True)

print('------------------------------ Finished dump of input info ----------------', flush=True)

lp = concurrent_core.get_local_paths(df)

print('Location paths=' + str(lp))

print('------------------------------ Begin Loading Huggingface ner model ------------------', flush=True)
try:
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
except Exception as err:
    print('Caught ' + str(err) + ' while loading ner model')
print('------------------------------ After Loading Huggingface ner model ------------------', flush=True)

print('------------------------------ Begin Creating Huggingface ner pipeline ------------------', flush=True)
ner = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
print('------------------------------ After Creating Huggingface ner pipeline ------------------', flush=True)

def do_ner_fnx(row):
    print("do_ner_fnx: Entered. row=" + str(row))
    s = ner(row['text'])[0]
    orgs = []
    persons = []
    misc = []
    for entry in s:
        print("do_ner_fnx: Entry=" + str(entry))
    print("do_ner_fnx: Exit. returning orgs=" + str(orgs) + ", persons=" + str(persons) + ", misc=" + str(misc)) 
    return [orgs, persons, misc]

print('------------------------------ Before Inference ------------------', flush=True)
consolidated_pd = None
for one_local_path in lp:
    print('Begin processing file ' + str(one_local_path), flush=True)
    df1 = pd.read_pickle(one_local_path)
    if consolidated_pd:
        consolidated_pd = pd.DataFrame(df1)
    else:
        consolidated_pd = df1

consolidated_pd.reset_index()
for index, row in consolidated_pd.iterrows():
    consolidated_pd[['orgs', 'persons', 'misc']] = consolidated_pd.apply(do_ner_fnx, axis=1, result_type='expand')
    print("'" + row['text'] + "' sentiment=" + row['label'] + ", score=" + str(row['score']) + ", orgs=" + str(row['orgs']) + ", persons=" + str(row['persons']) + ", misc=" + str(row['misc']))

tfname = "/tmp/output.pickle"
if os.path.exists(tfname):
    os.remove(tfname)
consolidated_pd.to_pickle(tfname)
concurrent_core.concurrent_log_artifact(tfname, "output.pickle")
print('Finished logging artifacts file')

print('------------------------------ After Inference. End ------------------', flush=True)

os._exit(os.EX_OK)
