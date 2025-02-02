import pandas as pd
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

class Elastic:
    @staticmethod
    def __index(index_name: str, series: pd.Series):
        body = series.to_dict()
        es.index(
            index=index_name,
            body = body
        )
    @staticmethod
    def index_movie(movie: pd.Series):    
        Elastic.__index("movies", movie)
    @staticmethod   
    def index_book(book: pd.Series):
        Elastic.__index("books", book)
    @staticmethod
    def index_music(music: pd.Series):
        Elastic.__index(index_name="music", series=music)

