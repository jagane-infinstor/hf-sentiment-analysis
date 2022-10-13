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

print('------------------------------ Begin Loading Huggingface ner model ------------------', flush=True)
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
print('------------------------------ After Loading Huggingface ner model ------------------', flush=True)

print('------------------------------ Begin Creating Huggingface ner pipeline ------------------', flush=True)
ner = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
print('------------------------------ After Creating Huggingface ner pipeline ------------------', flush=True)

def do_nlp_fnx(row):
    s = nlp(row['text'])[0]
    return [s['label'], s['score']]

def do_ner_fnx(row):
    s = ner(row['text'])[0]
    orgs = []
    persons = []
    misc = []
    for entry in s:
        if entry['entity_group'] == 'ORG':
            orgs.append(entry['word'])
        elif entry['entity_group'] == 'PER':
            persons.append(entry['word'])
        elif entry['entity_group'] == 'MISC':
            misc.append(entry['word'])
    return [orgs, persons, misc]

print('------------------------------ Before Inference ------------------', flush=True)
negatives = 0
positives = 0
for one_local_path in lp:
    print('Begin processing file ' + str(one_local_path), flush=True)
    jsonarray = pickle.load(open(one_local_path, 'rb'))
    # for i in jsonarray:
    #   print(json.dumps(i), flush=True)
    df1 = pd.DataFrame(jsonarray, columns=['text'])
    df1[['label', 'score']] = df1.apply(do_nlp_fnx, axis=1, result_type='expand')
    df1.reset_index()
    df1[['orgs', 'persons', 'misc']] = df1.apply(do_ner_fnx, axis=1, result_type='expand')
    df1.reset_index()
    for index, row in df1.iterrows():
        print("'" + row['text'] + "' sentiment=" + row['label'] + ", score=" + str(row['score']) + ", orgs=" + str(row['orgs']) + ", persons=" + str(row['persons']) + ", misc=" + str(row['misc']))
        if row['label'] == 'NEGATIVE' and row['score'] > 0.9:
            negatives = negatives + 1
        if row['label'] == 'POSITIVE' and row['score'] > 0.9:
            positives = positives + 1
    print('Finished processing file ' + str(one_local_path) + ': + ' + str(positives) + ', - ' + str(negatives), flush=True)
    # tf_fd, tfname = tempfile.mkstemp()
    # df1.to_pickle(tfname)
    # concurrent_core.concurrent_log_artifact(tfname, "result/" + os.path.basename(os.path.normpath(one_local_path)))
    # print('Finished logging artifacts file')

fn = '/tmp/sentiment_summary.json'
if os.path.exists(fn):
    os.remove(fn)
sentiment_summary = {'positives': positives, 'negatives': negatives}
with open(fn, 'w') as f:
    f.write(json.dumps(sentiment_summary))
concurrent_core.concurrent_log_artifact(fn, "")

print('------------------------------ After Inference. End ------------------', flush=True)

os._exit(os.EX_OK)
