import logging
import pandas as pd
from utilities.vectorization import Vectorization

logger = logging.getLogger(__name__)

def clean_columns(df: pd.DataFrame, verbose: bool = True):
    if verbose: logger.info("Cleaning columns...")
    df.columns = ['index', 'overview', 'release_date', 'vote_average', 'title', 'genres']

def clean_rows(df: pd.DataFrame, verbose: bool = True):
    if verbose: logger.info("Cleaning rows...")
    df['overview_vector'] = Vectorization.word2vec(corpus=df['overview'])
    df['title_vector'] = Vectorization.word2vec(corpus=df["title"])

def main(verbose: bool = True):
    movies = pd.read_csv("./data/original/movies.csv")
    clean_columns(df=movies, verbose=verbose)
    clean_rows(df=movies, verbose=verbose)
    if verbose: logger.info("Exporting data...")
    movies.to_csv("./data/cleaned/movies.csv")

if __name__ == "__main__":
    main()