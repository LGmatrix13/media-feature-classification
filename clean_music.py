import logging
import re
from utilities.vectorization import Vectorization
import concurrent.futures
import pandas as pd

logger = logging.getLogger(__name__)

def clean_rows(df: pd.DataFrame, verbose: bool = True):
    if verbose: logger.info("Cleaning rows...")
    def clean_lyrics(lyrics: str) -> str: 
        lyrics = re.sub('\n', ' ', lyrics)
        lyrics = re.sub(r"\[.*?\]", "", lyrics).strip()
        return lyrics
    
    df = df[df['language'] == 'en']
    df['features'] = df['features'].apply(lambda row: row.replace("{", '').replace("}", '').split(","))
    df['lyrics'] = df['lyrics'].apply(clean_lyrics)
    df['lyrics_vector'] = df['lyrics'].apply(Vectorization.tfidf)

def main(verbose: bool = True):
    def process(df: pd.DataFrame, file: str):
        clean_rows(df=df, verbose=verbose)
        if verbose: logger.info("Exporting file...")
        df.to_csv(file)

    with concurrent.futures.ProcessPoolExecutor() as executor:    
        executor.map(
            [
                process(df=chunk, file=f"./data/cleaned/music_{index}.csv") 
                for index, chunk in enumerate(pd.read_csv("./data/original/music.csv", chunksize=100000))
            ]
        )

if __name__ == "__main__":
    main()
