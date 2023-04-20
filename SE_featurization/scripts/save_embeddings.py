import argparse
import warnings

from tqdm import tqdm
import pandas as pd
import torch
import sys

from moses.models_storage import ModelsStorage

warnings.filterwarnings("ignore", category=UserWarning)


def get_collate_fn(model):
    def create_tensors(string, device):
        regression_values = string[2:]
        return [
            model.string2tensor(string[0], device=torch.device('cuda')),
            model.string2tensor(string[1], device=torch.device('cuda')),
            regression_values,
        ]

    def collate(data):
        sort_fn = lambda x: len(x[0])
        # data.sort(key=sort_fn, reverse=True)
        tensors = [create_tensors(string, torch.device('cuda')) for string in data]
        tensors.sort(key=sort_fn, reverse=True)
        return tensors

    return collate


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="A file containing a list of smiles",
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Embeddings created from the spectrum",
    )

    parser.add_argument(
        "--model_load",
        type=str,
        required=True,
        help="Where to load the model",
    )
    parser.add_argument(
        "--config_load",
        type=str,
        required=True,
        help="Where to load the config",
    )

    parser.add_argument(
        "--vocab_load",
        type=str,
        required=True,
        help="Where to load the vocab",
    )
    parser.add_argument(
        "--device", type=str, required=True, help="cuda/cpu?",
    )

    parser.add_argument(
        "--n_batch", type=int, required=True, help="Batch Size",
    )

    parser.add_argument("--model", type=str, required=True, help="Model name")

    return parser


def main(model, config):
    #debugpy()
    device = torch.device(config.device)

    # For CUDNN to work properly:
    if device.type.startswith("cuda"):
        torch.cuda.set_device(device.index or 0)
    MODELS = ModelsStorage()

    model_config = torch.load(config.config_load)
    model_vocab = torch.load(config.vocab_load)
    model_state = torch.load(config.model_load)
    
    print("Model loaded.")
    trainer = MODELS.get_model_trainer(model)(config)

    model = MODELS.get_model_class(model)(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    print("Calculating Embeddings..")
    df = pd.read_csv(config.input_file, sep='\t',names=['SMILES'])
    #df = pd.read_csv('/dccstor/trustedgen/data/pubchem/CID-SMILES-CANONICAL-10000.smi', sep='\t',names=['SMILES'])
    smiles = df['SMILES'].tolist()
    data = [smiles, smiles, smiles]
    data = list(zip(*data))
    train_loader = tqdm(trainer.get_dataloader(model, data, shuffle=False, collate_fn=get_collate_fn(model)))

    smiles_embeddings = dict()
    with torch.no_grad():
        for i, input_batch in enumerate(train_loader):
            randomized_smiles, canonical_smiles, smiles = zip(
                *input_batch
            )
            mu, logvar, z, kl_loss, recon_loss = model(
                    randomized_smiles, canonical_smiles
            )
            for j in range(len(randomized_smiles)):
                smiles_embeddings[smiles[j][0]] = mu[j].cpu()

    torch.save(smiles_embeddings, config.output_file, _use_new_zipfile_serialization=False)
    

if __name__ == '__main__':
    parser = get_parser()
    print(parser)
    config = parser.parse_args()
    print(config)
    model = sys.argv[1]
    main('translation', config)
