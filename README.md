## Accurate  Clinical Toxicity Prediction using Multi-task Deep Neural Nets and Contrastive Molecular Explanations

Provided here is the code to: 
- create multi-task DNN and single-task DNN models to predict <i> in vivo </i> (RTECS), <i> in vitro </i> (Tox21), and clinical (ClinTox) toxicity given molecular input (Morgan fingerprints, FP, and SMILES Embeddings, SE)
- compute a novel molecular representation - SMILES Embeddings
- extract contrastive molecular explanations for toxicity predictions 

## Prerequisites 

The scripts were run in an anaconda environment with the list of packages and versions provided in the "anaconda_environment_packages.csv" file. 

Follow the installation instructions in: https://github.com/Trusted-AI/AIX360, to download packages needed for Contrastive Explanations Method

Jupyter notebooks is needed to run the deep predictive models provided.

##  Data 

<b> Raw Data </b>: Tox21 (<i> in vitro </i>), and ClinTox (clinical) datasets from MoleculeNet are in the "data/datasets/raw_data" folder 
- ClinTox source: https://github.com/deepchem/deepchem/tree/master/examples/clintox/datasets
- Tox21 source: MoleculeNet https://github.com/deepchem/deepchem/blob/7463d93d0f85a3ba58cd155209540d8e649d875e/deepchem/molnet/load_function/tox21_datasets.py uses https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz

<b> Split Data </b>: Train / valid / test sets for 122, 123, 124 seeds is given in the "data/datasets/split_data" folder for Tox21 and ClinTox
- Data was split using the seeds and random splitting method from MoleculeNet. The process to obtain the splits are provided in the "data/datasets/obtaining_splits" folder

<b> RTECS </b>(<i> in vivo </i>) data is commerical by Biovia (available for a fee or by way of subscription), and cannot be provided. However, the code using this dataset is given. 

##  Modeling 

#### SMILES Embedding (SE) Featurization

Code to create SE featurization is provided in "SE_featurization" folder. More details are given at "SE_featurization/scripts/README.md".

Dictionary of mapping SMILES in Tox21, ClinTox, and RTECS dataset to a SMILES Embedding is given in "data/smiles_embedding/toxicity_smiles.zip". Unzip file before use.

### Deep Predictive Models

For deep predictive models, Jupyter notebooks provide the process to: 
 - load data
 - featurize the input (FP, SE)
 - create train/valid/test sets for DNNs
 - build the MTDNN model
 - train the model 
 - evaluate performance on test set using different performance metrics

### <i> Deep Learning </i>

##### Multi-task DNN (MTDNN)

Within the "deep_predictive_models/deep_learning" folder are Jupyter Notebooks to create pytorch multi-task DNN models  of different combinations of datasets (Tox21, ClinTox, RTECS), using inputs of:
- FP ("deep_predictive_models/deep_learning/FP/MTDNN") = MTDNN-FP
- SE ("deep_predictive_models/deep_learning/SE/MTDNN") = MTDNN-SE

##### Single-task DNN (STDNN)

Within the "deep_predictive_models/deep_learning" folder are Jupyter Notebooks to create pytorch single-task DNN models for Tox21, ClinTox and RTECS, using inputs of:
- FP ("deep_predictive_models/deep_learning/FP/STDNN") = STDNN-FP
- SE ("deep_predictive_models/deep_learning/SE/STDNN") = STDNN-SE

### <i> Transfer Learning </i>

Under "deep_predictive_models/transfer_learning" folder are Jupyter Notebooks to create and test transfer learning models using different combinations of Tox21 (<i> in vitro </i>) and RTECS (<i> in vivo </i>) as base models, transferred to predict clinical toxicity (ClinTox). 

### Contrastive Explanations Method (CEM)

#### Obtaining explanations

"cem/cem_explanations" folder supplies python scripts to obtain pertinent positives (PP) and pertinent negatives (PN) as explanations of single-task DNN predictions on Tox21, ClinTox and RTECS.

The scripts returns dataframes with information on the input molecules, predictions, true labels, and PP and PNs. Each row is added to the dataframe according to weight - with higher weighted explanations added first. 

The top ten PP and PNs were obtained by calculating the most frequent (by count) PP and PNs for correctly predicted (true label = predicted label) molecules. Any conflicts rising from same counts within the top ten ranks, were resolved by averaging the weight rank order the PP/PN was added to the dataframe. 

#### Base trained STDNN model

Jupyter notebooks in the "cem/trained_model" folder train and test the keras STDNN models that are explained by the CEM, for predictions on <i> in vitro </i> (Tox21), <i> in vivo </i> (RTECS), and clinical (ClinTox) toxicity. 

### Genetic Algorithm (GA) Features 

Under the "genetic_algorithm" folder are scripts to obtain a set of near optimal input (as Morgan fingerprint bits) to single-task toxicity prediction models on Tox21, ClinTox and RTECS. 

Each run of the GA, even given the same parameters, will not return the same results since the GA is a probabilistic stochastic search. Our obtained results (bits chosen to be near optimal) for Tox21, ClinTox and RTECS are given in the "genetic_algorithm/results" folder. 

The bits obtained were matched to the bits of the PP and PNs, to examine if the top ten PP and PNs are within the optimal set of input features specified by the GA.
