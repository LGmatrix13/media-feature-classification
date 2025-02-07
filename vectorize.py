from argparse import ArgumentParser, Namespace
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd

def main(args: Namespace):
    input_df = pd.read_csv(args.input_file)
    output_df = pd.DataFrame(columns=["index", "vector"])
    model = Doc2Vec(
        vector_size=10,
        alpha=0.025,
        min_alpha=0.00025,
        min_count=1,
        dm=1
    )
    for index, row in input_df.iterrows():
        tokenized = word_tokenize(text=row[args.text_column])
        vector = model.infer_vector(tokenized)
        input_df = pd.concat(input_df, pd.Series({
            "index": index,
            "vector": vector
        }))
    output_df.to_csv(args.file)

    
if __name__ == "__main__":
    parser = ArgumentParser(description=(
        "Run a vectorization process on a given dataset"
    ))
    parser.add_argument('input_file', type=str, help='path to input dataset')
    parser.add_argument('text_column', type=str, help='column name to dataset')
    parser.add_argument('output_file', type=str, help='path to export output dataset')
    args = parser.parse_args()
    main(args)