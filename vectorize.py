from argparse import ArgumentParser, Namespace
import nltk
import pandas as pd
import gensim.downloader as api 
import numpy as np
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def embed(text: str, model):
    words = word_tokenize(text)
    vectors = [model[word] for word in words if word in model]
    if vectors:
        return np.mean(vectors, axis=0).tolist()
    return None

def main(args: Namespace):
    input_df = pd.read_parquet(args.input_file)
    model = api.load("word2vec-google-news-300")
    output_data = []
    
    for index, row in input_df.iterrows():
        print(f"Processing row {index}...")
        if row[args.text_column] is None:
            output_data.append({ "pk": row[args.pk_columnn], "vector": None })
        else:
            vector = embed(row[args.text_column], model)
            output_data.append({ "pk": row[args.pk_columnn], "vector": vector })

    output_df = pd.DataFrame(output_data)
    output_df.to_parquet(args.output_file, index=False)

if __name__ == "__main__":
    parser = ArgumentParser(description="Run a vectorization process on a given dataset using a pretrained Word2Vec model.")
    parser.add_argument("input_file", type=str, help="Path to input dataset")
    parser.add_argument("pk_columnn", type=str, help="id/primary key of the input dataset")
    parser.add_argument("text_column", type=str, help="Column name containing text data")
    parser.add_argument("output_file", type=str, help="Path to export output dataset")
    args = parser.parse_args()
    main(args)