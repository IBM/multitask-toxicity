import pandas as pd
import fire

def main(input_file, output_file):
    df = pd.read_csv(input_file, names=['SMILES'])
    nrows = len(df)
    df['ID'] = list(range(nrows))
    df.to_csv(output_file, header=True, index=False, columns=['ID','SMILES'])

if __name__=='__main__':
    fire.Fire(main)



