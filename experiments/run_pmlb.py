import numpy as np
import pandas as pd
import tqdm
import glob
import re
from experiment import Experiment

#Import all datasets
datasets = glob.glob('/path_to_pmlb_datasets/*.csv')
LOGS_PATH =  '/path_to_logs/'
EXPERIMENT_PREFIX = 'pmlb_'
TARGET_NAME = 'target'
TRAIN_SIZE = 0.7
dfs = []
for dataset in datasets:
    item = {}
    item['name'] = dataset.split("/")[-1]
    item['data'] = pd.read_csv(dataset)
    dfs.append(item)
print("{} datasets loaded".format(len(dfs)))

experiments_done = [x.split("/")[-1] for x in glob.glob(LOGS_PATH+"/"+EXPERIMENT_PREFIX+"*.csv")]

#Run experiment
for i in range(len(dfs)):
    item = dfs[i]
    exp_name = EXPERIMENT_PREFIX+'_'+item['name']
    if exp_name not in experiments_done:
        print("Starting experiment {}".format(exp_name))
        experiment = Experiment(item['data'], TARGET_NAME, exp_name, LOGS_PATH, TRAIN_SIZE)
        experiment.run()
    else:
        print("Skipping experiment {}, since it was found at log directory".format(exp_name))
print("Run successfull!")
