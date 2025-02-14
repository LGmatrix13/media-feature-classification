import pandas as pd

def chunk_movies():
    chunksize = 10 ** 4
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
        overview_chunk = chunk.loc[english_only, ['overview']]
        overview = pd.concat([overview, overview_chunk])
        chunk_idx += 1

    metadata.to_parquet('./data/raw/movies_metadata.parquet')
    overview.to_parquet('./data/raw/movies_overview.parquet')

def chunk_music():
    chunksize = 10 ** 6
    chunk_idx = 0
    metadata = pd.DataFrame()
    lyrics = pd.DataFrame()

    for chunk in pd.read_csv('./data/raw/music.csv', chunksize=chunksize, usecols=['title', 'tag', 'artist', 'year', 'views', 'features', 'lyrics', 'id', 'language_cld3']):
        print(f"Running chunk {chunk_idx}...")
        english_only = chunk['language_cld3'] == 'en'
        metadata_chunk = chunk.loc[english_only, ['title', 'tag', 'artist', 'year', 'views', 'features']]
        metadata = pd.concat([metadata, metadata_chunk])
        lyrics_chunk = chunk.loc[english_only, ['lyrics','language_cld3']]
        lyrics = pd.concat([lyrics, lyrics_chunk])
        chunk_idx += 1

    metadata.to_parquet('./data/raw/music_metadata.parquet')
    lyrics.to_parquet('./data/raw/music_lyrics_metadata.parquet')

if __name__ == "__main__":
    main()