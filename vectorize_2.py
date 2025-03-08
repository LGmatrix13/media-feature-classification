import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

TYPE = "classification"
FILE = "./data/raw/movies_overview.parquet"
FREETEXT_COLUMN = "overview"  # Make sure this column exists in your dataset
DB_NAME = "./data/vectors/movies_vectors.db"
TABLE_NAME = "embeddings"

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

def create_table():
    """Creates the SQLite table if it does not exist."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY,
                vector TEXT
            )
        ''')
        conn.commit()

def process_data_in_chunks(df: pd.DataFrame, chunk_size=70):
    """Processes data in chunks and inserts embeddings into SQLite."""
    num_rows = len(df)
    total_chunks = (num_rows + chunk_size - 1) // chunk_size 
    for chunk_idx, i in enumerate(range(0, num_rows, chunk_size), start=1):
        print(f"Processing chunk {chunk_idx}/{total_chunks} ({i} to {min(i+chunk_size, num_rows)})")
        chunk = df.iloc[i:i + chunk_size]
        embeddings = generate_embeddings(df=chunk)
        insert_into_db(chunk, embeddings)

def generate_embeddings(df: pd.DataFrame):
    """Generates embeddings for the given DataFrame."""
    sentences = [f'{TYPE}: {freetext}' for freetext in df[FREETEXT_COLUMN].astype(str).values]
    embeddings = model.encode(sentences)
    return embeddings

def insert_into_db(df: pd.DataFrame, embeddings: np.ndarray):
    """Inserts embeddings into the SQLite database."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        for idx, embedding in zip(df.index, embeddings):
            cursor.execute(f'''
                INSERT INTO {TABLE_NAME} (id, vector)
                VALUES (?, ?)
            ''', (int(idx), np.array2string(embedding, separator=',')))
        conn.commit()

# Create table
create_table()

if __name__ == "__main__":
    df = pd.read_parquet(FILE)
    process_data_in_chunks(df=df, chunk_size=70)
