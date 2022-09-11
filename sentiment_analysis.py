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
from parallels_plugin import parallels_core

print('sentiment_analysis: Entered', flush=True)
df = parallels_core.list(None, input_name='tweets')
print('sentiment_analysis: After parallels_core.list', flush=True)

print('Column Names:', flush=True)
cn = df.columns.values.tolist()
print(str(cn))

for ind, row in df.iterrows():
    print('index=' + str(ind) + ", row=" + str(row), flush=True)

os._exit(os.EX_OK)

#print(str(sys.argv))

## tdir = tempfile.mkdtemp()
## print('model directory=' + str(tdir))
## ModelsArtifactRepository("models:/HFSentimentAnalysis/Production").download_artifacts(artifact_path="", dst_path=tdir)
## model = mlflow.pyfunc.load_model(tdir)
## print('model=' + str(model))

#inp = ['This is great weather', 'This is terrible weather']
#jsonarray = pickle.load(open('/home/jagane/Downloads/1565264790192365568', 'rb'))
#for i in jsonarray:
#  print(json.dumps(i))

#df = pd.DataFrame(jsonarray, columns=['text'])
#ii = model.predict(df)
#ii.reset_index()
#for index, row in ii.iterrows():
#  print("'" + row['text'] + "' sentiment=" + row['label'] + ", score=" + str(row['score']))
