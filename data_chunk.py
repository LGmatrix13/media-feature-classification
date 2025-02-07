import pandas as pd

def main():
    chunksize = 10 ** 6
    chunk_idx = 0
    for chunk in pd.read_csv('./data/raw/music.csv', chunksize=chunksize, usecols=['title', 'tag', 'artist', 'year', 'views', 'features', 'lyrics', 'id', 'language_cld3']):
        print(f"Running chunk {chunk_idx}...")
        chunk[chunk['language_cld3'] == 'en'].to_parquet(f'./data/raw/music_{chunk_idx}.parquet')
        chunk_idx += 1

if __name__ == "main":
    main()