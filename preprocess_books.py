from bs4 import BeautifulSoup
import pandas as pd


def preprocess(df: pd.DataFrame):
    df[['author_1', 'author_2']] = df['authors'].str.split("/", n=2, expand=True)
    df = df.drop(columns=["language", "authors", 'isbn'])
    df['total_authors'] = df.apply(lambda row: sum(pd.notna(row[col]) for col in ['author_1', 'author_2']), axis=1)
    return df