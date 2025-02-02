from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

tfidf_vectorizer = TfidfVectorizer()

class Vectorization():
    @staticmethod
    def tfidf(corpus: str) -> list[list[float]]:
        X = tfidf_vectorizer.fit_transform(corpus)
        return X.toarray()
    
    @staticmethod
    def word2vec(corpus: str) -> list[list[float]]:
        sentences = [sentence.split(" ") for sentence in corpus.split(".")]
        return [[]]

