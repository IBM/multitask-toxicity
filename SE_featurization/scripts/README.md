# SMILES Embeddings

This code generates the embeddings for a molecule using a translation model by translating from a non-canonical SMILES to canonical SMILES.

The embeddings for the molecules used in this paper have been generated and stored at data/smiles_embedding

To generate the embeddings for the molecules, run

```
python -m scripts.save_embeddings --output_file out.pt --input_file smiles.csv --model translation --device cuda --model_load models/model.pt --vocab_load models/vocab.nb --config_load models/config.nb --n_batch 65
```

`output_file`: The generated embedings will be saved in this file. It is a pytorch file that can be loaded using `torch.load` and is a dictionary with the SMILES string as key and the embeddings as value
`input_file`: The input file containing SMILES, one per line.
`model` : The specific model identifier. Should always be ‘translation’
`device`: cuda or cpu. cuda is recommended
`model_load`: Path to the model
`vocab_load`: Path to the vocabulary
`config_load`: Path to the model config
`n_batch`: Batch size


Code adapted from MOSES repository: https://github.com/molecularsets/moses
The model is similar to the one developed in Winter et. al. Learning continuous and data-driven molecular descriptors by translating equivalent chemical representations. DOI: https://doi.org/10.1039/C8SC04175J 
