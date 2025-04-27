from argparse import ArgumentParser
from model_base import GenreClassifier

if __name__ == "__main__":
    parser = ArgumentParser(description="Train a multi-class movie genre classifier")
    parser.add_argument('n', type=int, help='Number of samples')
    args = parser.parse_args()
    
    classifier = GenreClassifier(
        data_path='./data/transformed/music_lyrics.parquet',
        metadata_path="./data/transformed/music_metadata.parquet",
        paratext_col="lyrics",
        genre_col="tag",
        n = args.n
    )
    classifier.train()