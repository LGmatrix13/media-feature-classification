from argparse import ArgumentParser, Namespace
import pickle
import gensim
import nltk
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize    
import plotly.express as px

def embed(paratext: str, gensim_model) -> np.ndarray:
    """embed a paratext using gensim"""

    print("Embedding query...")
    words = word_tokenize(paratext.lower())
    vectors = [gensim_model[word] for word in words if word in gensim_model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300) 

def train(data, type: str) -> tuple[str, any]:
    """train a clustering model"""

    if type == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=500, metric="cosine")
        labels = model.fit_predict(data)
        return labels, model
    elif type == "kmeans":     
        normalized_data = normalize(data, norm='l2')
        model = KMeans(n_clusters=4, random_state=42)
        labels = model.fit_predict(normalized_data)
        return labels, model
    else:
        raise ValueError("Invalid type")

def books(type: str, sample_size: int = None, verbose: bool = True) -> tuple[str, any]:
    """prepare data and train model for books"""

    if verbose: print(f"Reading books data...")
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
    if verbose: print(f"Training {type} on bookss data...")
    labels, model = train(data=data, type=type)
    return labels, model, data

def movies(type: str, sample_size: int = None, verbose: bool = True) -> tuple[str, any]:
    """prepare data and train model for movies"""

    if verbose: print(f"Reading movies data...")
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
    if verbose: print(f"Training {type} on movies data...")
    labels, model = train(data=data, type=type)
    return labels, model, data

def music(type: str, sample_size: int = None, verbose: bool = True) -> tuple[str, any]:
    """prepare data and train model for music"""

    if verbose: print("Reading music data...")
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
    if verbose: print(f"Training {type} on movies data...")
    labels, model = train(data=data, type=type)
    return labels, model, data

def predict(model: KMeans, vector: np.ndarray) -> int:
    """prepare data and train model for music"""
    if isinstance(model, KMeans):
        return model.predict([vector])[0] + 1

def report(data, labels, verbose: bool = True):
    """report characteristics of the clustering"""

    if verbose: print("Running PCA to reduce vector deminisons of the data...")

    pca = PCA(n_components=3)  # Reduce to 2D for visualization
    reduced_data = pca.fit_transform(data)
    fig = px.scatter_3d(reduced_data, x=0, y=1, z=2, color=labels, title="Cluster Visualization")

    if verbose: print("Generating distribution data of each cluster...", end="\n\n")
    unique_clusters = np.unique(labels)
    if len(unique_clusters) > 1:
        score = silhouette_score(data, labels)
        print(f"Silhouette Score: {score:.4f}")
    else:
        print("Silhouette Score: Not applicable (only one cluster found).")
    for cluster in unique_clusters:
        cluster_indices = np.where(labels == cluster)[0]
        print(f"Cluster {cluster + 1}: {len(cluster_indices)} points")
    
    fig.show()


def read(medium: str, type: str, sample_size: int) -> tuple[any, any]:
    """read a cached model"""

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
    verbose = args.verbose
    paratext = args.paratext

    nltk.download('punkt')
    MODEL_PATH = "./models/word2vec-google-news-300.model"  # Path to the pre-saved .npy file
    gensim_model = gensim.models.keyedvectors.KeyedVectors.load(MODEL_PATH)

    cached = read(medium=medium, type=type, sample_size=sample_size)
    if cached is not None:        
        vector = embed(paratext=paratext, gensim_model=gensim_model) if len(paratext) > 0 else None
        del gensim_model
        labels, model, data = cached
        report(data=data, labels=labels)
        if len(paratext) > 0: 
            prediction = predict(model=model, vector=vector)
            print(f"Predicted cluster: {prediction}")
        return
    
    if medium == "movies":
        vector = embed(paratext=paratext, gensim_model=gensim_model) if len(paratext) > 0 else None
        del gensim_model
        labels, model, data = movies(type=type, sample_size=sample_size)
        report(data=data, labels=labels, verbose=verbose)
        if vector is not None: 
            prediction = predict(model=model, vector=vector)
            print(f"Predicted cluster: {prediction}")
        export(training=(labels,model, data), medium=medium, type=type, sample_size=sample_size)
    elif medium == "music":
        vector = embed(paratext=paratext, gensim_model=gensim_model) if len(paratext) > 0 else None
        del gensim_model
        labels, model, data = music(type=type, sample_size=sample_size)
        report(data=data, labels=labels, verbose=verbose)
        if vector is not None: 
            prediction = predict(model=model, vector=vector)
            print(f"Predicted cluster: {prediction}")
        export(training=(labels,model, data), medium=medium, type=type, sample_size=sample_size)
    elif medium == "books":
        vector = embed(paratext=paratext, gensim_model=gensim_model) if len(paratext) > 0 else None
        del gensim_model
        labels, model, data = books(type=type, sample_size=sample_size)
        report(data=data, labels=labels, verbose=verbose)
        if vector is not None: 
            prediction = predict(model=model, vector=vector)
            print(f"Predicted cluster: {prediction}")
        export(training=(labels,model, data), medium=medium, type=type, sample_size=sample_size)
    else:
        raise ValueError("Invalid medium")

if __name__ == "__main__":
    parser = ArgumentParser(description=(
        "Specify media medium to cluster on.",
        "Specify the type of model.",
        "Specify the sample size to train on.",
        "Specify whether to enable verbose reporting.",
        "Provide custom paratext to classify. Will only classify with kmeans."
    ))
    parser.add_argument('medium', type=str, choices=["movies", "music", "books"], help='medium type')
    parser.add_argument('type', type=str, choices=["dbscan", "kmeans"], help='type of model')
    parser.add_argument('sample_size', type=int, help='size of sample')
    parser.add_argument('verbose', nargs='?', type=bool, choices=[True, False], default=True, help="specify verbose printing for each stage")
    parser.add_argument('paratext', nargs='?', type=str, default='', help="specify verbose printing for each stage")

    args = parser.parse_args()
    main(args)