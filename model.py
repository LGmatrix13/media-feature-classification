from argparse import ArgumentParser, Namespace
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def train(data, type: str) -> tuple[str, any]:
    if type == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(data)
        return labels, model
    elif type == "kmeans":       
        model = KMeans(n_clusters=4, random_state=42)
        labels = model.fit_predict(data)
        return labels, model
    elif type == "hdbscan":
        model = model.HDBSCAN(min_cluster_size=10)  # Adjust this parameter as needed
        labels = model.fit_predict(data)
        return labels, model
    else:
        raise ValueError("Invalid type")

def books(type: str, sample_size: int = None) -> tuple[str, any]:
    vectors = pd.read_parquet("./data/vectors/books_descriptions_vectors.parquet")
    vectors.columns = ["id", "vector"]
    vectors = vectors[~vectors['vector'].isnull()]
    metadata = pd.read_parquet("./data/transformed/books_metadata.parquet")
    metadata = metadata.sample(n=sample_size if sample_size is not None else len(metadata))
    metadata = metadata[['id', 'name', 'author_1']]
    df = vectors.merge(metadata, on="id")
    del vectors
    del metadata
    data = np.vstack(df['vector'].values)
    labels, model = train(data=data, type=type)
    return labels, model

def movies(type: str, sample_size: int = None) -> tuple[str, any]:
    vectors = pd.read_parquet("./data/vectors/movies_overview_vectors.parquet")
    vectors.columns = ["id", "vector"]
    vectors = vectors[~vectors['vector'].isnull()]
    metadata = pd.read_parquet("./data/transformed/movies_metadata.parquet")
    metadata = metadata.sample(n=sample_size if sample_size is not None else len(metadata))
    metadata = metadata[['id', 'title', 'production_company_1']]
    df = vectors.merge(metadata, on="id")
    del vectors
    del metadata
    data = np.vstack(df['vector'].values)
    labels, model = train(data=data, type=type)
    return labels, model, data

def music(type: str, sample_size: int = None) -> tuple[str, any]:
    vectors = pd.read_parquet("./data/vectors/music_lyrics_vectors.parquet")
    vectors.columns = ["id", "vector"]
    vectors = vectors[~vectors['vector'].isnull()]
    metadata = pd.read_parquet("./data/transformed/music_metadata.parquet")
    metadata = metadata.sample(n=sample_size if sample_size is not None else len(metadata))
    metadata = metadata[['id', 'title', 'artist']]
    df = vectors.merge(metadata, on="id")
    del vectors
    del metadata
    data = np.vstack(df['vector'].values)
    labels, model = train(data=data, type=type)
    return labels, model, data

def report(data, labels):
    pca = PCA(n_components=2)  # Reduce to 2D for visualization
    reduced_data = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("KMeans Clustering Visualization (PCA Reduced)")
    plt.show()

    unique_clusters = np.unique(labels)
    print("Cluster Report:")
    for cluster in unique_clusters:
        cluster_indices = np.where(labels == cluster)[0]
        print(f"Cluster {cluster}: {len(cluster_indices)} points")

def read(medium: str, type: str, sample_size: int) -> tuple[any, any]:
    try:
        with open(f'./models/model_{medium}_{type}_{sample_size}.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    
def export(training: tuple[any, any, any], medium: str, type: str, sample_size: int):
    with open(f'./models/model_{medium}_{type}_{sample_size}.pkl', 'wb') as f:
        pickle.dump(training, f)

def main(args: Namespace):
    type = args.type
    medium = args.medium
    sample_size = args.sample_size
    
    cached = read(medium=medium, type=type, sample_size=sample_size)

    if cached is not None:
        labels, model, data = cached
        report(data=data, labels=labels)
        return

    if medium == "movies":
        labels, model, data = movies(type=type, sample_size=sample_size)
        report(data=data, labels=labels)
        export(training=(labels,model, data), medium=medium, type=type, sample_size=sample_size)
    elif medium == "music":
        labels, model, data = music(type=type, sample_size=sample_size)
        report(data=data, labels=labels)
        export(training=(labels,model, data), medium=medium, type=type, sample_size=sample_size)
    elif medium == "books":
        labels, model, data = books(type=type, sample_size=sample_size)
        report(data=data, labels=labels)
        export(training=(labels,model, data), medium=medium, type=type, sample_size=sample_size)
    else:
        raise ValueError("Invalid medium")

if __name__ == "__main__":
    parser = ArgumentParser(description=(
        "Specify the type of model",
        "Specify media medium to cluster on",
        "Specify the sample size to train on"
    ))
    parser.add_argument('medium', type=str, choices=["movies", "music", "books"], help='path to JSON config file with proceedures')
    parser.add_argument('type', type=str, choices=["dbscan", "kmeans", "hbscan"], help='path to JSON config file with proceedures')
    parser.add_argument('sample_size', type=int, help='path to JSON config file with proceedures')
    args = parser.parse_args()
    main(args)