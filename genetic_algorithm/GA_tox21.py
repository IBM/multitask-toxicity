#!/usr/bin/env python
# coding: utf-8

# In[1]:

##################### IMPORTS #######################
import os
import sys
from keras.models import model_from_json
from matplotlib import pyplot as plt
import numpy as np

from keras.callbacks import ModelCheckpoint
from rdkit.Chem.Draw import IPythonConsole

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

with HiddenPrints():
    print("This will not be printed")

print("This will be printed as before")

# general and data handling
import numpy as np
import pandas as pd
import os
from collections import Counter

# Required RDKit modules
import rdkit as rd
from rdkit import DataStructs
from rdkit.Chem import AllChem
import rdkit.Chem.MCS

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

# modeling
import sklearn as sk
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
from torch.utils.data import Dataset, DataLoader
import time
import random
import joblib

##################### SETTINGS + DATA #######################
''' Note, before use:
         - define "filepath" variable in "DEEP NEURAL NETWORK" section, the path to the trained keras single-task DNN model 
                  with the matching given architecture used for CEM explanations
         - define "explain_task" variable, an int value between 0 and 11, for the task to be explained in Tox21: 
                 ['NR-AR', 'NR-Aromatase', 'NR-PPAR-gamma', 'SR-HSE', 
                   'NR-AR-LBD', 'NR-ER', 'SR-ARE', 'SR-MMP',
                   'NR-AhR', 'NR-ER-LBD', 'SR-ATAD5', 'SR-p53']
         - To save the explanation results, define the pathway in the "SAVE RESULTS" section
'''

# set seed
seed_value = 122 #122 123 124, as used in MoleculeNet
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

# number of bits for morgan fingerprints
morgan_bits = 4096

# number of radius for morgan fingerprints
morgan_radius = 2 

#GA hyperparameters
generations = 50
features = 100
population = 100

# set the Tox21 task, choose a value between 0 and 11 for each of the tasks in tox21_tasks
# can be an argument for this file as well with 
#    explain_task = int(sys.argv[1])
explain_task = 0

# raw dataset
tox21_file  = '../data/tox21/tox21.csv'
tox21_tasks = ['NR-AR', 'NR-Aromatase', 'NR-PPAR-gamma', 'SR-HSE', 
               'NR-AR-LBD', 'NR-ER', 'SR-ARE', 'SR-MMP',
               'NR-AhR', 'NR-ER-LBD', 'SR-ATAD5', 'SR-p53']
tox21_data = pd.read_csv(tox21_file)
data = [tox21_data]

# Setting task and labels
all_tasks = tox21_tasks 
task = all_tasks[explain_task]
task_label = task

# load saved tox21 train/test/valid data 
data_path = "../data/datasets/tox21/split_data/seed_122/"
train_data=torch.load(data_path + 'train_data_tox21.pth')
test_data=torch.load(data_path + 'test_data_tox21.pth')
valid_data=torch.load(data_path + 'valid_data_tox21.pth')

data = [train_data, test_data, valid_data]

# construct morgan fingerprints 
for i in range(len(data)):
    data[i]['mol'] = [rd.Chem.MolFromSmiles(x) for x in data[i]['smiles']]

    bi = [{} for _ in range(len(data[i]))]
    data[i]['morgan'] = [AllChem.GetMorganFingerprintAsBitVect(data[i].iloc[j]['mol'], morgan_radius, nBits = morgan_bits, bitInfo=bi[j]) for j in range(len(data[i]))]
    data[i]['bitInfo'] = bi

data = [train_data, test_data, valid_data]


# replace NA with -1 -- used to deal with missing labels, along with Binary Cross-Entropy loss
data[0] = data[0].fillna(-1)
data[1] = data[1].fillna(-1)
data[2] = data[2].fillna(-1)

train_data = data[0]
test_data  = data[1]
valid_data = data[2]


## Arrays for train / test / valid sets used for DNN (convert the RDKit explicit vectors into numpy arrays)
# Train
x_train = []
for fp in train_data['morgan']:
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    x_train.append(arr)
x_train = np.array(x_train)
#x_train = x_train - 0.5

y_train = train_data[all_tasks].astype('int').values

# Test
x_test = []
for fp in test_data['morgan']:
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    x_test.append(arr)
x_test = np.array(x_test)
#x_test = x_test - 0.5

y_test = test_data[all_tasks].astype('int').values

# Valid
x_valid = []
for fp in valid_data['morgan']:
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    x_valid.append(arr)
x_valid = np.array(x_valid)
#x_valid = x_valid - 0.5

y_valid = valid_data[all_tasks].astype('int').values

##################### DEEP NEURAL NETWORK #######################
''' Defines a single-task DNN for the specified task (explain_task) 
    Same architecture as the single-task DNN created in pytorch 
'''
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras.layers import LeakyReLU
from keras.utils import to_categorical

input_shape = morgan_bits

input_fp = Input(shape=(input_shape,))

task_output = [None for task in all_tasks]
for i in range(len(all_tasks)):    
    hidden_task = Dense(1024)(input_fp)
    hidden_task = BatchNormalization()(hidden_task)
    hidden_task = LeakyReLU(alpha=0.05)(hidden_task)
    
    hidden_task = Dense(512)(hidden_task)
    hidden_task = BatchNormalization()(hidden_task)
    hidden_task = LeakyReLU(alpha=0.05)(hidden_task)

    task_output[i] = Dense(2, activation='softmax', name=all_tasks[i])(hidden_task)

deepnn = Model(input_fp, task_output)

deepnn.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

### Load trained model 
''' Use pathway to trained model, and load it into the deepnn
    '''
filepath=#"use-trained-model-pathway/checkpoint.hdf5"

## Load trained weights into model - only once the model pathway has been defined
deepnn.load_weights(filepath)

## Predict
y_test_pred = deepnn.predict(x_test)

###################### SET UP GA ######################

from genetic_selection import GeneticSelectionCV
from sklearn.ensemble import RandomForestClassifier

# the sklearn-genetic package needs sklearn estimators
estimator = RandomForestClassifier()

model = GeneticSelectionCV(
    estimator, cv=5, verbose=1,
    scoring="accuracy", max_features=features,
    n_population=population, crossover_proba=0.5,
    mutation_proba=0.2, n_generations=generations,
    crossover_independent_proba=0.5,
    mutation_independent_proba=0.04,
    tournament_size=3, n_gen_no_change=10,
    caching=True, n_jobs=-1)


##################### OBTAIN GA EXPLANATIONS ######################

# Define specific task for GA
all_tasks = all_tasks[explain_task]

# Dictionary to store results 
GA_bits = {all_tasks : None}

# Labels for train set for specific task
y_train_task = y_train[:,explain_task] 

# Carry out GA
model = model.fit(x_train, y_train_task)

# Index in x_train set refers to a certain bit, obtain this information
# This defines the optimal set of bits to use as input for this prediction
target_bits = [i for i, x in enumerate(model.support_) if x]

# Update results dictionary
GA_bits[task_label] = target_bits 


##################### SAVE GA EXPLANATIONS ######################

import pickle 

# Store data (serialize)
with open('GA_tox21.pickle', 'wb') as handle:
    pickle.dump(GA_bits, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print("Generation scores:", model.generation_scores_)
print("Num. of selected features:", model.n_features_)