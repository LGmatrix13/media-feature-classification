import logging
import nltk
import pandas as pd
import gensim.downloader as api

nltk.download('punkt_tab')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Vectorization:  
    @staticmethod
    def word2vec(corpus: pd.Series, verbose: bool = True) -> pd.Series:
        if verbose: logger.info("Starting vectorization...")
        values = corpus.astype(str).values
        if verbose: logger.info("Tokenizing rows...")
        tokenized_sentences = [nltk.word_tokenize(value) for value in values]
        if verbose: logger.info("Training model...")
        wv = api.load('word2vec-google-news-300')
        if verbose: logger.info("Assigning vectors to each row...")
        vectors = pd.Series([
            [wv[word] for word in sentence if word in wv]  # List of word vectors
            for sentence in tokenized_sentences
        ])
        return vectors
