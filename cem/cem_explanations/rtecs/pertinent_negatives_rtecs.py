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
         - RTECS dataset is commerical, thus could not be provided. Instead given is the code without defining the dataset 
                  path
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
# (cannot provide since it is a commerical dataset)
a_oral_file = # cannot provide
a_oral_data = pd.read_csv(a_oral_file)
a_oral_tasks = ['toxic_a_oral'] 

# setting task and labels
task = a_oral_tasks[0]
task_label = a_oral_tasks[0]
data = [a_oral_data]
all_tasks = a_oral_tasks

# load saved rtecs split train/test/valid data 
# (datapath is not provided, since RTECS is a commericial dataset and cannot be shared)
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

#positives = pd.DataFrame(columns = ['index', 'smiles', 'pp', 'bit']) 

# dataframe to save the PNs
negatives = pd.DataFrame(columns = ['index', 'smiles', 'molecule', 'predicted_class', 'predicted_prob', 'true_class', 'pn', 'pn_structure', 'bit', 'from_mol_smiles', 'from_mol_m']) 

# as a check, for any indices (chemicals) that did not work in obtaining PP 
bad_index = pd.DataFrame(columns = ['index'])

## Settings for CEM

arg_max_iter = 1000   # Maximum number of iterations to search for the optimal PN for given parameter settings
arg_init_const = 10.0 # Initial coefficient value for main loss term that encourages class change
arg_b = 9             # No. of updates to the coefficient of the main loss term

arg_kappa = 0.01   # Minimum confidence gap between the PNs (changed) class probability and original class' probability
arg_beta = 0.99      # Controls sparsity of the solution (L1 loss)
arg_gamma = 0   # Controls how much to adhere to a (optionally trained) autoencoder

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


##################### OBTAIN PERTINENT NEGATIVE (PN) EXPLANATION ######################

# Maximum number of explanations
max_explanation = 10

'''
     - For each index in the test set (each chemical), the CEM generates PNs.      
     - To accelerte the process, this script had been passed as a job to a computing cluster, computing PNs of the test set 
            within sets of (start_index, end_index), e.g. (0, 300). The results were concatenated for the full test set. 
            However, the PNs for the entire test set can be computed in one job (shown below), given computing power and time. 
     - start_index and end_index can be defined as arguments into the script, as:
            start_index = int(sys.argv[1])
            end_index = int(sys.argv[2])
     - For each PN, the predicted class, predicted probability, true class, rdkit mol, SMILES of the molecule, the SMILES,  
            bit, rdkit substructure image of the PP, and SMILES, rdkit mol of the molecule the bit belongs to, is added as a  
            new row in the dataframe
     - Each PN is added to the dataframe according to weight -- with higher weighted PNs added first
'''

for i in range(len(test_data)):
#for i in range(start_index, end_index): 
    if i < len(test_data): 
            try: 
                test_input = x_test[[i]]
                task = task_label
                
                # generate pertinent negative (explanation) from CEM
                print('Optimizing for pertinent negative...')
                with HiddenPrints():
                    arg_mode = "PN" 
                    (adv_pn, delta_pn, info_pn) = explainer.explain_instance(test_input, arg_mode, ae_model, arg_kappa, arg_b, 
                                                                arg_max_iter, arg_init_const, arg_beta, arg_gamma)

                mol, bi = test_data.iloc[i][['mol', 'bitInfo']]

                predicted_class = mymodel.predict_classes(test_input)[0]
                pred_proba = mymodel.predict(test_input)[:,predicted_class][0]
                true_class = test_data.iloc[i][task]

                pos_bits, pos_weights, neg_bits, neg_weights = get_bits(-delta_pn)

                for j in range(np.min([max_explanation, len(pos_bits)])):
                    target_bit = pos_bits[j]
                    weight = pos_weights[j]
                    df_ex = train_data[x_train[:,target_bit] == 1]
                    df_ex = df_ex[df_ex[task] != predicted_class]
                    if len(df_ex) > 0:
                        
                        ### Obtain information on molecule the PN is taken from
                        s, m, b, c = df_ex.sample(1)[['SMILES', 'mol', 'bitInfo', task]].iloc[0]
                        
                        ### Obtain SMILES of the PN given the target bit 
                        atom_center = b[target_bit][0][0]
                        bit_radius = b[target_bit][0][1]
                        
                        env = rd.Chem.FindAtomEnvironmentOfRadiusN(m,bit_radius,atom_center)
                        amap={}
                        submol=rd.Chem.PathToSubmol(m,env,atomMap=amap)
                        
                        # bit_radius = 0 is a PN that is defined only as the atom 
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
                        mfp2_svg = rd.Chem.Draw.DrawMorganBit(m, target_bit, b, useSVG=True)
                        newrow = {'index': i, 'smiles': test_data.iloc[i]['SMILES'], 'molecule': mol, 
                                  'predicted_class': predicted_class, 
                                  'predicted_prob': pred_proba, 'true_class': true_class,
                                  'pn': bit_smiles, 'pn_structure': mfp2_svg, 'bit': target_bit, 
                                  'from_mol_smiles': rd.Chem.MolToSmiles(m, True), 'from_mol_m': m
                        }
                        negatives = negatives.append(newrow, ignore_index = True)

            except:
                # Save information on any indices (chemicals) for which explanations were not obtained
                index_row = {'index': i}
                bad_index.append(index_row, ignore_index = True)
                continue

##################### SAVE RESULTS ######################  

## Define the path wanted for the results 
#negatives.to_pickle('path-to-results/pn_rtecs.csv')

## Save any indices for which explanations were not found --- however when checked resulted into any empty dataframe
#bad_index.to_csv('path-to-results/bad_index_pn_rtecs.csv', index=False)