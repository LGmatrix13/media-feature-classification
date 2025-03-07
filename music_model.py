from argparse import ArgumentParser
import gensim
from matplotlib import pyplot as plt
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def load_data():
    try:
        print("Loading cached data...")
        df_vectors = pd.read_pickle("./data/raw/music_lyrics_vectors.pkl")
        df_metadata = pd.read_pickle("./data/raw/music_metadata.pkl")
    except FileNotFoundError:
        print("Pickle not found. Reading from Parquet (slow)...")
        df_vectors = pd.read_parquet("./data/raw/music_lyrics_vectors.parquet", columns=["pk", "vector"])
        df_metadata = pd.read_parquet("./data/raw/music_metadata.parquet", columns=['id', "tag"])   
        df_vectors.to_pickle("./data/raw/music_lyrics_vectors.pkl")
        df_metadata.to_pickle("./data/raw/music_metadata.pkl")
    
    df_vectors.columns = ["id", "vector"]
    return df_vectors, df_metadata
    
def embed(paratext: str, model) -> np.ndarray:
    print("Embedding query...")
    words = word_tokenize(paratext.lower())
    vectors = [model[word] for word in words if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300) 

def train_svm(df_vectors: pd.DataFrame, df_metadata: pd.DataFrame, feature: str, verbose: bool = True) -> DBSCAN:
    print("Training SVM model...")
    df_vectors = df_vectors[~df_vectors["vector"].isnull()].sample(n=100000)
    merged = df_vectors.merge(df_metadata, on="id")
    X = np.vstack(merged[feature])
    y = merged[feature].valuess
    del merged
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    if verbose: print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    if verbose: _visualize_clusters(X, y)   
    return svm

def train_knn(df_vectors: pd.DataFrame, df_metadata: pd.DataFrame, feature: str, verbose: bool = True) -> tuple[KNeighborsClassifier, list]:
    if verbose: print("Training model...")
    df_vectors = df_vectors[~df_vectors["vector"].isnull()].sample(n=100000)
    merged = df_vectors.merge(df_metadata, on="id")
    X = np.vstack(merged[feature])
    y = merged[feature].valuess
    del merged
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)   
    knn.fit(X_train, y_train)   
    y_pred = knn.predict(X_test)
    if verbose: print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")  
    if verbose: _visualize_clusters(X, y)   
    return knn

# Report results
def report(model, df_metadata: pd.DataFrame, embedding: np.ndarray, feature: str):
#    indexes = model.kneighbors([embedding], n_neighbors=20, return_distance=False)
    
 #   print("Nearest neighbors:")
  #  for index in indexes[0]:
   #     music = df_metadata.iloc[index]
    #    print(f"{music['title']}, {music[feature]}")  # Or any other column from df_metadata

    print(f"Prediction for {feature}: {model.predict([embedding])[0]}")

def _visualize_clusters(X, labels):
    print("Visualizing clusters with t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    colors = sns.color_palette("hsv", len(unique_labels))
    
    for label, _ in zip(unique_labels, colors):
        plt.scatter(X_2d[labels == label, 0], X_2d[labels == label, 1], label=f"Class {label}", alpha=0.6, edgecolors='k')
    
    plt.legend()
    plt.title("t-SNE Visualization of SVM Classifications")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

# Main function
def main():
    parser = ArgumentParser(description="Run the music classification/clustering model.")
    parser.add_argument("paratext", type=str, help="Paratext to use for prediction")
    parser.add_argument("feature", type=str, help="Feature to predict")
    parser.add_argument("knn", type=bool, help="Use KNN. Otherwise it will use SVM")
    args = parser.parse_args()

    nltk.download('punkt', quiet=True)
    MODEL_PATH = "./models/word2vec-google-news-300.model"  # Path to the pre-saved .npy file
    model = gensim.models.keyedvectors.KeyedVectors.load(MODEL_PATH)
    df_vectors, df_metadata = load_data()
    embedding = embed(args.paratext, model)
    del model
    if args.knn:
        trained_model = train_knn(df_vectors=df_vectors, df_metadata=df_metadata, feature=args.feature)
    else:
        trained_model = train_svm(df_vectors=df_vectors, df_metadata=df_metadata, feature=args.feature)
    del df_vectors
    report(model=trained_model, df_metadata=df_metadata, embedding=embedding, feature=args.feature)
   
if __name__ == "__main__":
    main()
