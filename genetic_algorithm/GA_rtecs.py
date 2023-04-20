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

# GA hyperparameters
generations = 45
features = 300
population = 100

# raw dataset
a_oral_file = # cannot provide
a_oral_data = pd.read_csv(a_oral_file)
a_oral_tasks = ['toxic_a_oral']

# Setting task and labels
task = a_oral_tasks[0]
task_label = a_oral_tasks[0]
data = [a_oral_data]
all_tasks = a_oral_tasks

# load saved rtecs split train/test/valid data 
data_path = #cannot provide
train_data=torch.load(data_path + 'train_data_rtecs.pth')
test_data=torch.load(data_path + 'test_data_rtecs.pth')
valid_data=torch.load(data_path + 'valid_data_rtecs.pth')

data = [train_data, test_data, valid_data]

# construct morgan fingerprints 
for i in range(len(data)):
    data[i]['mol'] = [rd.Chem.MolFromSmiles(x) for x in data[i]['smiles']]

    bi = [{} for _ in range(len(data[i]))]
    data[i]['morgan'] = [AllChem.GetMorganFingerprintAsBitVect(data[i].iloc[j]['mol'], morgan_radius, nBits = morgan_bits, bitInfo=bi[j]) for j in range(len(data[i]))]
    data[i]['bitInfo'] = bi

# replace NA with -1  -- used to deal with missing labels, along with Binary Cross-Entropy loss
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

y_train = train_data[task].values

# Test
x_test = []
for fp in test_data['morgan']:
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    x_test.append(arr)
x_test = np.array(x_test)

y_test = test_data[task].values

# Valid
x_valid = []
for fp in valid_data['morgan']:
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    x_valid.append(arr)
x_valid = np.array(x_valid)

y_valid = valid_data[task].values

##################### DEEP NEURAL NETWORK #######################
''' Defines a single-task DNN for the specified task  
    Same architecture as the single-task DNN created in pytorch 
'''
import keras
from keras.layers import Input, Dense, Activation, LeakyReLU
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

input_shape = x_train.shape[1]

deepnn = Sequential([
    Dense(2048, input_shape=(input_shape,)),
    Activation('relu'),
    Dense(1024),
    Activation('relu'),
    Dense(512),
    Activation('relu'),
    Dense(2),
    Activation('softmax'),
])

deepnn.compile(optimizer='adam', loss='binary_crossentropy')

### Load trained model 
''' Use pathway to trained model, and load it into the deepnn
    '''
filepath=#"use-trained-model-pathway/checkpoint.hdf5"

## Load trained weights into model - only once the model pathway has been defined
deepnn.load_weights(filepath)

y_test_pred = deepnn.predict(x_test)[:,1]

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

model = model.fit(x_train, y_train)

# Index in x_train set refers to a certain bit, obtain this information
# This defines the optimal set of bits to use as input for this prediction
target_bits = [i for i, x in enumerate(model.support_) if x]


##################### SAVE GA EXPLANATIONS ######################


# Store data (serialize)
with open('GA_rtecs.pickle', 'wb') as handle:
    pickle.dump(target_bits, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print("Generation scores:", model.generation_scores_)
print("Num. of selected features:", model.n_features_)