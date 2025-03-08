import os
import sqlite3
from google import genai
from google.genai.types import EmbedContentConfig
import pandas as pd
import time
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['GOOGLE_AI_API_KEY']
client = genai.Client(api_key=api_key)

df = pd.read_parquet("./data/raw/movies_overview.parquet")
df = df.loc[~df['overview'].isnull()].reset_index(drop=True)  # Reset index for consistency

# Set up SQLite database
conn = sqlite3.connect("./data/vectors/movies_overview_vectors.sqlite")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS vectors (
        id INTEGER PRIMARY KEY,
        vector TEXT
    )
""")
conn.commit()

def vectorize_in_batches(batch_size, max_retries=15):
    contents = df[['id', 'overview']].values.tolist()
    
    try:
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            batch_ids, batch_texts = zip(*batch)
            print(f"Processing batch {i//batch_size + 1}/{(len(contents) - 1) // batch_size + 1} with {len(batch_texts)} items")
            
            retries = 0
            while retries < max_retries:
                try:
                    result = client.models.embed_content(
                        model="text-embedding-004",
                        contents=batch_texts,
                        config=EmbedContentConfig(output_dimensionality=384, task_type="CLASSIFICATION"),
                    )
                    
                    # Write embeddings to SQLite line by line
                    for idx, embedding in zip(batch_ids, result.embeddings):
                        cursor.execute("INSERT INTO vectors (id, vector) VALUES (?, ?)", (idx, str(embedding.values)))
                    conn.commit()
                    
                    time.sleep(0.5)  # Avoid API rate limiting
                    break  # Success, exit retry loop

                except Exception as e:
                    retries += 1
                    print(f"Error in batch {i//batch_size + 1}: {e}")
                    if retries < max_retries:
                        print(f"Retrying batch {i//batch_size + 1} (attempt {retries}/{max_retries}) after sleep...")
                        time.sleep(30)
                    else:
                        print(f"Failed after {max_retries} retries. Moving to the next batch.")
    except:
        print("Something wrong happened. Exiting early.")
        conn.close()
        exit(1)

vectorize_in_batches(batch_size=85)
conn.close()
print("Vectorization complete and saved to SQLite.")
