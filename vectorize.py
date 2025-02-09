from argparse import ArgumentParser, Namespace
import pandas as pd
import ollama

def embed(text: str):
    return ollama.embeddings(model='all-minilm', prompt=text).get('embedding')

def main(args: Namespace):
    input_df = pd.read_parquet(args.input_file)
    output_data = []
    
    for index, row in input_df.iterrows():
        print(f"Processing row {index}")
        if row[args.text_column] is None:
            output_data.append({"index": index, "vector": None})  # Convert numpy array to list
        else:
            vector = embed(row[args.text_column])
            output_data.append({"index": index, "vector": vector})

    # Create output DataFrame and save
    output_df = pd.DataFrame(output_data)
    output_df.to_parquet(args.output_file, index=False)

if __name__ == "__main__":
    parser = ArgumentParser(description="Run a vectorization process on a given dataset using a pretrained Doc2Vec model.")
    parser.add_argument("input_file", type=str, help="Path to input dataset")
    parser.add_argument("text_column", type=str, help="Column name containing text data")
    parser.add_argument("model_path", type=str, help="Path to the pretrained Doc2Vec model")
    parser.add_argument("output_file", type=str, help="Path to export output dataset")
    args = parser.parse_args()
    main(args)
