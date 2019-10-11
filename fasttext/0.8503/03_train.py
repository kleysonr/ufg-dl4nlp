import mlflow
import fasttext
import time
import os
from snsdl.keras.wrappers.base_model import BaseModel

start_time = time.time()

os.makedirs('model', exist_ok=True)

# # Space search
# paramsSearch = {
#     'minCount': [1],
#     'minCountLabel': [0],
#     'wordNgrams': [1, 2],
#     'bucket': [2000000],
#     'minn': [3],
#     'maxn': [5],
#     't': [0.0001],
#     'lr': [0.5, 0.55],
#     'lrUpdateRate': [100],
#     'dim': [100, 300],
#     'ws': [5],
#     'epoch': [50],
#     'neg': [5],
#     'loss': ['ns'],
#     'thread': [12],
#     'seed': [40],
#     'input': ['output/train_portuguese.txt'],
# }

# Best Score
#
# Starting training 5/8
# {'minCount': 1, 'minCountLabel': 0, 'wordNgrams': 2, 'bucket': 2000000, 'minn': 3, 'maxn': 5, 't': 0.0001, 'lr': 0.5, 'lrUpdateRate': 100, 'dim': 100, 'ws': 5, 'epoch': 50, 'neg': 5, 'loss': 'ns', 'thread': 12, 'seed': 40, 'input': 'output/train_portuguese.txt'}
# Read 77M words
# Number of words:  402788
# Number of labels: 1576
# Progress: 100.0% words/sec/thread:  129586 lr:  0.000000 avg.loss:  0.102325 ETA:   0h 0m 0s
# MLflow Run ID: 27c6ce3e9934411b96e4637ef9d97029
# Starting test for @1
# Starting test for @5

paramsSearch = {
    'minCount': [1],
    'minCountLabel': [0],
    'wordNgrams': [2],
    'bucket': [2000000],
    'minn': [3],
    'maxn': [5],
    't': [0.0001],
    'lr': [0.5],
    'lrUpdateRate': [100],
    'dim': [100],
    'ws': [5],
    'epoch': [50],
    'neg': [5],
    'loss': ['hs'],
    'thread': [12],
    'seed': [40],
    'input': ['output/train_spanish.txt'],
}

myModel = BaseModel(paramsSearch)
params = myModel.getSearchParams()

for i, p in enumerate(params):
 
    p_start_time = time.time()

    print('Starting training {}/{}'.format(i+1, len(params)))
    print(p)

    model = fasttext.train_supervised(**p)

    model.save_model('model/model.bin'.format(p_start_time))

    with mlflow.start_run():
 
        # print out current run_uuid
        run_uuid = mlflow.active_run().info.run_uuid
        print("MLflow Run ID: %s" % run_uuid)
 
        # log parameters
        for k, v in p.items():
            mlflow.log_param(k, v)
 
        print('Starting test for @1')
        (nsamples, precision, recall) = model.test('output/validation_spanish.txt')

        mlflow.log_metric('nsamples', nsamples)
        mlflow.log_metric('precision_1', precision)
        mlflow.log_metric('recall_1', recall)

        print('Starting test for @5')
        (nsamples, precision, recall) = model.test('output/validation_spanish.txt', k=5)

        mlflow.log_metric('precision_5', precision)
        mlflow.log_metric('recall_5', recall)

        mlflow.log_artifacts('model', 'model')


end_time = time.time()
hours, rem = divmod(end_time-start_time, 3600)
minutes, seconds = divmod(rem, 60)
print('Finished in {:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))