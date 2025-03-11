import pandas as pd

def preprocess(df: pd.DataFrame):
    df = df.drop_duplicates(subset=["id"], keep='first')
    df = df[df['title'] != 'Untitled']
    df[['keyword_1', 'keyword_2', 'keyword_3', 'keyword_4', 'rest']] = df['keywords'].str.split(',', n=4, expand=True)
    df = df[df['status'] != 'Canceled']
    df[['genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_rest']] = df['genres'].str.split(',', n=4, expand=True)
    df[['production_company_1', 'production_company_2', 'production_company_3', 'production_company_4', 'production_company_rest']] = df['production_companies'].str.split(',', n=4, expand=True)
    df = df.drop(columns=["production_company_rest", "genre_rest", "rest", "spoken_languages", "production_companies", "genres", "poster_path", "original_title", "original_language", "imdb_id", "homepage", "backdrop_path"])
    df[['production_country_1', 'production_country_2', 'production_country_3', 'production_country_rest']] = df['production_countries'].str.split(',', n=3, expand=True)
    df = df.drop(columns=["production_country_rest", "keywords", "production_countries"])
    df['total_keywords'] = df.apply(lambda row: sum(pd.notna(row[col]) for col in ['keyword_1', 'keyword_2', 'keyword_3', 'keyword_4']), axis=1)
    df['total_genre'] = df.apply(lambda row: sum(pd.notna(row[col]) for col in ['genre_1', 'genre_2', 'genre_3', 'genre_4']), axis=1)
    df['total_production_company'] = df.apply(lambda row: sum(pd.notna(row[col]) for col in ['production_company_1', 'production_company_2', 'production_company_3', 'production_company_4']), axis=1)
    df['total_production_country'] = df.apply(lambda row: sum(pd.notna(row[col]) for col in ['production_country_1', 'production_country_2', 'production_country_3']), axis=1)
    return df