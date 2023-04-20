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

# Graphing
import matplotlib.pyplot as plt

# To set seed values 
import torch
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
from torch.utils.data import Dataset, DataLoader
import time
import random
import joblib

##################### SETTINGS + DATA #######################
''' Note, before use:
         - define "filepath" variable in "DEEP NEURAL NETWORK" section, the path to the trained keras single-task DNN model 
                  with the matching given architecture
         - To save the explanation results, define the pathway in the "SAVE RESULTS" section
'''

# set seed value
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

# raw dataset
clintox_file = '../../../data/datasets/clintox/raw_data/clintox.csv'
clintox_task = ['CT_TOX']
clintox_data = pd.read_csv(clintox_file)

# Setting task and labels
task = clintox_task
task_label = 'CT_TOX'
data = [clintox_data]

# load saved clintox train/test/valid data 
data_path = "../../../data/datasets/clintox/split_data/seed_122/"
train_data=torch.load(data_path + 'train_data_clintox.pth')
test_data=torch.load(data_path + 'test_data_clintox.pth')
valid_data=torch.load(data_path + 'valid_data_clintox.pth')

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

y_train = train_data[task[0]].astype('int').values

# Test
x_test = []
for fp in test_data['morgan']:
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    x_test.append(arr)
x_test = np.array(x_test)

y_test = test_data[task[0]].astype('int').values

# Valid
x_valid = []
for fp in valid_data['morgan']:
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    x_valid.append(arr)
x_valid = np.array(x_valid)

y_valid = valid_data[task[0]].astype('int').values

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
    Dense(512, input_shape=(input_shape,)),
    LeakyReLU(alpha=0.05),
    Dense(256),
    LeakyReLU(alpha=0.05),
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

### Predict
y_test_pred = deepnn.predict(x_test)[:,1]


##################### LOAD TRAINED CONVOLUTIONAL AUTOENCODER MODEL #######################

from aix360.algorithms.contrastive import CEMExplainer, KerasClassifier

''' specify ae model -- however we are not not adhering to this model when obtaining explanations
                        as arg_gamma is set to 0 (below)
'''

# specify ae model
input_img = Input(shape=(input_shape,))
encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(input_shape, activation='sigmoid')(decoded)

ae_model = Model(input_img, decoded)

###################### SET UP CEM ######################

#### Initialize CEM Explainer to explain model predictions
# wrap deepnn into a framework independent class structure
mymodel = KerasClassifier(deepnn)

# initialize explainer object
explainer = CEMExplainer(mymodel)

# dataframe to save the PPs
positives = pd.DataFrame(columns = ['index', 'smiles', 'molecule', 'predicted_class', 'predicted_prob', 'true_class', 'pp', 'pp_structure', 'target_bit', 'bits']) 
 
# as a check, for any indices (chemicals) that did not work in obtaining PP 
bad_index = pd.DataFrame(columns = ['index'])

## Settings for CEM (PP)

arg_max_iter = 1000   # Maximum number of iterations to search for the optimal PP for given parameter settings
arg_init_const = 10.0 # Initial coefficient value for main loss term 
arg_b = 9             # No. of updates to the coefficient of the main loss term

arg_kappa = 0.1 
arg_beta = 0.01       # Controls sparsity of the solution (L1 loss)
arg_gamma = 0         # Controls how much to adhere to a (optionally trained) autoencoder


# return the explanations correspond to the added bits and removed bits 
def get_bits(delta_bits):
    delta = delta_bits[0]
    argsort = np.argsort(delta)
    sorted_delta = delta[argsort]
    
    positive_bits = argsort[sorted_delta > 0][::-1]
    positive_weig = sorted_delta[sorted_delta > 0][::-1]
    
    negative_bits = argsort[sorted_delta < 0]
    negative_weig = sorted_delta[sorted_delta < 0]
    
    return positive_bits, positive_weig, negative_bits, negative_weig


##################### OBTAIN PERTINENT POSITIVE (PP) EXPLANATION ######################

# Maximum number of explanations
max_explanation = 10

'''
     - For each index in the test set (each chemical), the CEM generates PPs. 
     - To accelerte the process, this script had been passed as a job to a computing cluster, computing PPs of the test set 
            within sets of (start_index, end_index), e.g. (0, 300). The results were concatenated for the full test set. 
            However, the PPs for the entire test set can be computed in one job (shown below), given computing power and time. 
     - start_index and end_index can be defined as arguments into the script, as:
            start_index = int(sys.argv[1])
            end_index = int(sys.argv[2])
     - For each PP, the predicted class, predicted probability, true class, bitInfo, SMILES of the molecule, and the SMILES,  
            bit, rdkit substructure image of the PP, is added as a new row in the dataframe
     - Each PP is added to the dataframe according to weight -- with higher weighted PPs added first
'''

for i in range(len(test_data)):
#for i in range(start_index, end_index): 
    if i < len(test_data): 
            try:
                test_input = x_test[[i]]
                task = task_label
                input_bits, _, _, _ = get_bits(test_input)

                # generate pertinent positive (explanation) from CEM
                print('Optimizing for pertinent positive...')
                with HiddenPrints():
                    arg_mode = "PP"
                    (adv_pp, delta_pp, info_pp) = explainer.explain_instance(test_input, arg_mode, ae_model, arg_kappa, arg_b, 
                                                                arg_max_iter, arg_init_const, arg_beta, arg_gamma)

                mol, bi = test_data.iloc[i][['mol', 'bitInfo']]

                predicted_class = mymodel.predict_classes(test_input)
                pred_proba = mymodel.predict(test_input)[:,predicted_class][0]
                true_class = test_data.iloc[i][task]

                pos_bits, pos_weights, neg_bits, neg_weights = get_bits(delta_pp)

                if len(pos_bits) > 0:
                    for j in range(np.min([max_explanation, len(pos_bits)])):
                        target_bit = pos_bits[j]
                        if target_bit in input_bits:
                            weight = pos_weights[j]
                            
                            ### Obtain SMILES of the PP given the target bit 
                            atom_center = bi[target_bit][0][0]
                            bit_radius = bi[target_bit][0][1]
                            env = rd.Chem.FindAtomEnvironmentOfRadiusN(mol,bit_radius,atom_center)
                            amap={}
                            submol=rd.Chem.PathToSubmol(mol,env,atomMap=amap)
                            
                            # bit_radius = 0 is a PP that is defined only as the atom 
                            if bit_radius != 0 :
                                try: 
                                    bit_smiles = rd.Chem.MolToSmiles(submol,rootedAtAtom=amap[atom_center],canonical=True)
                                except:
                                    continue
                            elif bit_radius == 0: 
                                try:
                                    bit_smiles = mol.GetAtomWithIdx(atom_center).GetSymbol()
                                except:
                                    continue                    
                                    
                            ### Save information and substructure image as new row in dataframe
                            ### Explanations added by weight, higher weighted explanations added first 
                            mfp2_svg = rd.Chem.Draw.DrawMorganBit(mol, target_bit, bi, useSVG=True)

                            newrow = {'index': i, 'smiles': test_data.iloc[i]['SMILES'], 'molecule': mol, 
                                  'predicted_class': predicted_class[0], 
                                  'predicted_prob': pred_proba[0], 'true_class': true_class,
                                  'pp': bit_smiles, 'pp_structure': mfp2_svg, 'target_bit': target_bit, 
                                      'bits':bi
                                    }
                            positives = positives.append(newrow, ignore_index = True)


            except:
                # Save information on any indices (chemicals) for which explanations were not obtained
                index_row = {'index': i}
                bad_index.append(index_row, ignore_index = True)
                continue
                
##################### SAVE RESULTS ######################

## Define the path wanted for the results                 
#positives.to_pickle('path-to-results/pp_clintox.csv')

## Save any indices for which explanations were not found --- however when checked resulted into any empty dataframe
#bad_index.to_csv('path-to-results/bad_index_pp_clintox.csv', index=False)

