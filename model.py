from sklearn.neighbors import NearestNeighbors
import pandas as pd

print("Reading data...")
df = pd.read_parquet("./data/raw/music_lyrics_vectors.parquet")
X = df['vector']
print("Training model...")
knn = NearestNeighbors(n_neighbors=3)
knn.fit(X)
print("Calculating distances...")
distances, indices = knn.kneighbors(X)
print("Distances:", distances)
print("Indices:", indices)