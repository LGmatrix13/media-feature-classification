import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def chunk_movies():
    chunksize = 10 ** 5
    chunk_idx = 0
    metadata = pd.DataFrame()
    overview = pd.DataFrame()
    columns = ['id', 'title', 'vote_average', 'vote_count', 'status', 'release_date',
       'revenue', 'runtime', 'adult', 'backdrop_path', 'budget', 'homepage',
       'imdb_id', 'original_language', 'original_title', 'overview',
       'popularity', 'poster_path', 'tagline', 'genres',
       'production_companies', 'production_countries', 'spoken_languages',
       'keywords']
    for chunk in pd.read_csv('./data/raw/movies.csv', chunksize=chunksize, usecols=columns):
        print(f"Running chunk {chunk_idx}...")
        english_only = chunk['original_language'] == 'en'
        metadata_columns = [
            'id', 'title', 'vote_average', 'vote_count', 'status', 'release_date',
            'revenue', 'runtime', 'adult', 'backdrop_path', 'budget', 'homepage',
            'imdb_id', 'original_language', 'original_title',
            'popularity', 'poster_path', 'tagline', 'genres',
            'production_companies', 'production_countries', 'spoken_languages',
            'keywords'
            ]
        metadata_chunk = chunk.loc[english_only, metadata_columns]
        metadata = pd.concat([metadata, metadata_chunk])
        overview_chunk = chunk.loc[english_only, ['id', 'overview']]
        overview = pd.concat([overview, overview_chunk])
        chunk_idx += 1

    metadata.to_parquet('./data/raw/movies_metadata.parquet')
    overview.to_parquet('./data/raw/movies_overview.parquet')

def chunk_music():
    chunksize = 10**6
    lyrics_path = "./data/raw/music_lyrics.parquet"
    metadata_path = "./data/raw/music_metadata.parquet"
    lyrics_writer = None
    metadata_writer = None

    ids = set()

    for chunk_idx, chunk in enumerate(pd.read_csv('./data/raw/music.csv', chunksize=chunksize, usecols=['lyrics', 'id', 'language_cld3'])):
        print(f"Processing lyrics chunk {chunk_idx}...")

        english_only = chunk['language_cld3'] == 'en'
        lyrics_chunk = chunk.loc[english_only, ['id', 'lyrics']]

        if not lyrics_chunk.empty:
            ids.update(lyrics_chunk['id'])

            table = pa.Table.from_pandas(lyrics_chunk)

            if lyrics_writer is None:
                lyrics_writer = pq.ParquetWriter(lyrics_path, table.schema)
                lyrics_writer.write_table(table)

    if lyrics_writer:
        lyrics_writer.close()

    for chunk_idx, chunk in enumerate(pd.read_csv('./data/raw/music.csv', chunksize=chunksize, usecols=['title', 'tag', 'artist', 'year', 'views', 'features', 'id', 'language_cld3'])):
        print(f"Processing metadata chunk {chunk_idx}...")

        english_only = chunk['language_cld3'] == 'en'
        metadata_chunk = chunk.loc[english_only, ['title', 'tag', 'artist', 'year', 'views', 'features', 'id']]
        metadata_chunk = metadata_chunk[metadata_chunk['id'].isin(ids)]

        if not metadata_chunk.empty:
            table = pa.Table.from_pandas(metadata_chunk)

            if metadata_writer is None:
                metadata_writer = pq.ParquetWriter(metadata_path, table.schema)

            metadata_writer.write_table(table)

    if metadata_writer:
        metadata_writer.close()

    print("Parquet writing completed.")

if __name__ == "__main__":
    chunk_music()