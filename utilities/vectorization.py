from gensim.models import Word2Vec
import nltk
import pandas as pd

nltk.download('punkt_tab')

class Vectorization:  
    @staticmethod
    def word2vec(corpus: pd.Series) -> pd.Series:
        values = corpus.astype(str).values
        tokenized_sentences = [nltk.word_tokenize(value) for value in values]
        model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
        vectors = pd.Series([
            [model.wv[word] for word in sentence if word in model.wv]  # List of word vectors
            for sentence in tokenized_sentences
        ])
        return vectors
