from argparse import ArgumentParser, Namespace
import nltk
import pandas as pd
import gensim.downloader as api
import numpy as np
import sqlite3
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pyarrow.parquet as pq
import os

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def compute_tfidf(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    return vectorizer, tfidf_matrix, vocab

def is_not_stopword(word):
    return word not in stop_words

def embed(text: str, model, vectorizer, tfidf_weights, vocab):
    words = word_tokenize(text.lower())
    words = [word for word in words if word in vocab and is_not_stopword(word)]
    
    vectors = []
    weights = []
    
    for word in words:
        if word in model:
            vectors.append(model[word])
            weights.append(tfidf_weights[vocab.tolist().index(word)])
    
    if vectors:
        weighted_vectors = np.average(vectors, axis=0, weights=weights)
        return weighted_vectors.tolist()
    return None

def main(args: Namespace):
    model = api.load("word2vec-google-news-300")
    parquet_file = pq.ParquetFile(args.input_file)
    
    # Load text data for TF-IDF computation
    all_texts = []
    for batch in parquet_file.iter_batches():
        batch_df = batch.to_pandas()
        all_texts.extend(batch_df[args.text_column].dropna().astype(str).tolist())
    
    vectorizer, tfidf_matrix, vocab = compute_tfidf(all_texts)
    
    # SQLite setup
    conn = sqlite3.connect(args.output_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            pk TEXT PRIMARY KEY,
            vector BLOB
        )
    """)
    
    chunksize = 10 ** 6
    chunk_idx = 0
    for batch in parquet_file.iter_batches(batch_size=chunksize):
        print(f"Processing chunk {chunk_idx}...")
        batch_df = batch.to_pandas()
        
        for index, row in batch_df.iterrows():
            print(f"Vectorizing row {index}...")
            text = row[args.text_column]
            pk = row[args.pk_column]
            
            if text is not None:
                vector = embed(text, model, vectorizer, tfidf_matrix[index].toarray()[0], vocab)
                if vector:
                    cursor.execute("INSERT INTO vectors (pk, vector) VALUES (?, ?)", (pk, sqlite3.Binary(np.array(vector).tobytes())))
        
        conn.commit()
        chunk_idx += 1
    
    conn.close()

if __name__ == "__main__":
    parser = ArgumentParser(description="Run a vectorization process using a pretrained Word2Vec model with TF-IDF weighting.")
    parser.add_argument("input_file", type=str, help="Path to input dataset")
    parser.add_argument("pk_column", type=str, help="Primary key column name")
    parser.add_argument("text_column", type=str, help="Column name containing text data")
    parser.add_argument("output_file", type=str, help="SQLite database file path")
    args = parser.parse_args()
    main(args)
