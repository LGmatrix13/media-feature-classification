from argparse import ArgumentParser, Namespace
from model_base import GenreClassifier

if __name__ == "__main__":
    parser = ArgumentParser(description="Train a multi-class movie genre classifier")
    parser.add_argument('n', type=int, help='Number of samples')
    args = parser.parse_args()
    
    classifier = GenreClassifier(
        data_path='./data/transformed/movies_overview.parquet',
        metadata_path="./data/transformed/movies_metadata.parquet",
        paratext_col="overview",
        genre_col="genre_1",
        n = args.n
    )
    classifier.train()