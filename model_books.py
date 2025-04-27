from argparse import ArgumentParser
from model_base import GenreClassifier

if __name__ == "__main__":
    parser = ArgumentParser(description="Train a multi-class movie genre classifier")
    parser.add_argument('n', type=int, help='Number of samples')
    args = parser.parse_args()
    
    classifier = GenreClassifier(
        data_path='./data/transformed/book_descriptions_2.parquet',
        metadata_path="./data/transformed/book_metadata_2.parquet",
        paratext_col="summary",
        genre_col="genre",
        n = args.n
    )
    classifier.train()