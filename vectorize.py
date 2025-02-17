from argparse import ArgumentParser, Namespace
import nltk
import pandas as pd
import gensim.downloader as api 
import numpy as np
from nltk.tokenize import word_tokenize
import pyarrow.parquet as pq
import os
nltk.download('punkt')

def embed(text: str, model):
    words = word_tokenize(text)
    vectors = [model[word] for word in words if word in model]
    if vectors:
        return np.mean(vectors, axis=0).tolist()
    return None

def main(args: Namespace):
    model = api.load("word2vec-google-news-300")
    
    chunksize = 10 ** 6
    chunk_idx = 0
    parquet_file = pq.ParquetFile(args.input_file)
    files = []
    for batch in parquet_file.iter_batches(batch_size=chunksize):
        print(f"Processing chunk {chunk_idx}...")
        output_data = []
        batch_df = batch.to_pandas()

        for index, row in batch_df.iterrows():
            print(f"Vectorizing row {index}...")

            if row[args.text_column] is None:
                output_data.append({ "pk": row[args.pk_columnn], "vector": None })
            else:
                vector = embed(row[args.text_column], model)
                output_data.append({ "pk": row[args.pk_columnn], "vector": vector })

        output_df = pd.DataFrame(output_data)
        output_df.to_parquet(f"./data/raw/vectorize_{chunk_idx}.parquet", index=False)
        files.append(f"./data/raw/vectorize_{chunk_idx}.parquet")
        chunk_idx += 1

    schema = pq.ParquetFile(files[0]).schema_arrow
    with pq.ParquetWriter(args.output_file, schema=schema) as writer:
        for file in files:
            writer.write_table(pq.read_table(file, schema=schema))
            os.remove(file)

if __name__ == "__main__":
    parser = ArgumentParser(description="Run a vectorization process on a given dataset using a pretrained Word2Vec model.")
    parser.add_argument("input_file", type=str, help="Path to input dataset")
    parser.add_argument("pk_columnn", type=str, help="id/primary key of the input dataset")
    parser.add_argument("text_column", type=str, help="Column name containing text data")
    parser.add_argument("output_file", type=str, help="Path to export output dataset")
    args = parser.parse_args()
    main(args)