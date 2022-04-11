import numpy as np
import pandas as pd
import tqdm
import glob
import re
from experiment import Experiment

#Import all synthetic datasets
datasets = glob.glob('./synthetic_datasets/output/*.csv')
LOGS_PATH =  '/path_to_logs/'
EXPERIMENT_PREFIX = 'synthetic_'
TARGET_NAME = 'y'
TRAIN_SIZE = 0.7
regex_nums = re.compile(r"\d+")
dfs = []
for dataset in datasets:
    df = pd.read_csv(dataset)
    dataset_params = [int(x) for x in regex_nums.findall(dataset)]
    header = {'N': dataset_params[0],
              'v': dataset_params[1],
              'l': dataset_params[2],
              'd': dataset_params[3],
              'r': dataset_params[4]/100,
              'e': dataset_params[5]/100,
              'name': dataset.split("/")[-1],
              'data': df}
    dfs.append(header)
print("{} datasets loaded".format(len(dfs)))

experiments_done = [x.split("/")[-1] for x in glob.glob(LOGS_PATH+"/"+EXPERIMENT_PREFIX+"*.csv")]

#Run experiment
for i in range(len(dfs)):
    item = dfs[i]
    exp_name = EXPERIMENT_PREFIX+'_'+item['name']
    if exp_name not in experiments_done:
        experiment_metadata = dict([(key, item[key]) for key in item.keys() if key!='data' and key!='name'])
        print("Starting experiment {}".format(exp_name))
        data =  item['data'].drop(columns=['type'])
        experiment = Experiment(data, TARGET_NAME, exp_name, LOGS_PATH, TRAIN_SIZE, metadata=experiment_metadata)
        experiment.run()
    else:
        print("Skipping experiment {}, since it was found at log directory".format(exp_name))
print("Run successfull!")
