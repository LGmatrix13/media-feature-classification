import pandas as pd
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

class Elastic:
    def __init__(self):
        Elastic.__create_index("movies")
        Elastic.__create_index("books")
        Elastic.__create_index("music")

    @staticmethod
    def __create_index(index_name: str):
        if not es.indices.exists(index=index_name):
            es.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "embedding": {"type": "dense_vector", "dims": 128} 
                        }
                    }
                }
            )
            
    @staticmethod
    def __index(index_name: str, series: pd.Series):
        body = series.to_dict()
        es.index(
            index=index_name,
            body = body
        )

    def index_movie(movie: pd.Series):    
        Elastic.__index("movies", movie)

    def index_book(book: pd.Series):
        Elastic.__index("books", book)

    def index_music(music: pd.Series):
        Elastic.__index(index_name="music", series=music)

