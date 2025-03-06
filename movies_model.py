import gensim
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.neighbors import KNeighborsClassifier

def load_data():
    try:
        print("Loading cached data...")
        df_vectors = pd.read_pickle("./data/raw/movies_overview_vectors.pkl")
        df_metadata = pd.read_pickle("./data/raw/movies_metadata.pkl")
    except FileNotFoundError:
        print("Pickle not found. Reading from Parquet (slow)...")
        df_vectors = pd.read_parquet("./data/raw/movies_overview_vectors.parquet", columns=["pk", "vector"])
        df_metadata = pd.read_parquet("./data/raw/movies_metadata.parquet", columns=['id', "adult"])   
        # Save as Pickle for fast future loads
        df_vectors.to_pickle("./data/raw/movies_overview_vectorss.pkl")
        df_metadata.to_pickle("./data/raw/movies_metadata.pkl")
    
    df_vectors.columns = ["id", "vector"]
    return df_vectors, df_metadata
    
# Fast word embedding function
def embed(paratext: str, model) -> np.ndarray:
    print("Embedding query...")
    words = word_tokenize(paratext.lower())  # Normalize case
    vectors = [model[word] for word in words if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)  # Handle empty cases

# Optimized filtering using NumPy
# Parallelized filtering using NumPy
def filter(df_vectors: pd.DataFrame, embedding: np.ndarray) -> pd.DataFrame:
    print("Filtering data...")
    df_vectors = df_vectors[~df_vectors["vector"].isnull()]
    vectors = np.vstack(df_vectors["vector"].values)
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(embedding)
    cosine_similarities = np.dot(vectors, embedding) / norms
    del vectors
    df_vectors.loc[df_vectors.index, "cosine_similarity"] = cosine_similarities
    return df_vectors.nlargest(10000, "cosine_similarity")[["id", "vector", "cosine_similarity"]]

# Train KNN and NN
def train(df_vectors: pd.DataFrame, df_metadata: pd.DataFrame) -> tuple[KNeighborsClassifier, list]:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("Training model...")
    merged = df_vectors.merge(df_metadata, on="id")
    X = np.vstack(merged['vector'])
    y = merged['adult'].values  # Convert to NumPy
    del merged
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)   
    knn.fit(X_train, y_train)   
    y_pred = knn.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")  
    return knn, X_train

# Report results
def report(model: KNeighborsClassifier, X_train: list[list[float]], embedding: np.ndarray):
    indexes = model.kneighbors([embedding], n_neighbors=20, return_distance=False)
    print(X_train[indexes].ravel())
    return model.predict([embedding])

# Main function
def main():
    nltk.download('punkt')
    MODEL_PATH = "./models/word2vec-google-news-300.model"  # Path to the pre-saved .npy file
    model = gensim.models.keyedvectors.KeyedVectors.load(MODEL_PATH)
    df_vectors, df_metadata = load_data()
    query = input("Query: ")
    embedding = embed(query, model)
    del model
    filtered_df = filter(df_vectors, embedding)  # Parallel filtering
    del df_vectors
    trained_model, X_train = train(filtered_df, df_metadata)
    del df_metadata
    del filtered_df
    result = report(trained_model, X_train, embedding)
    print("Predicted Tag:", result)

if __name__ == "__main__":
    main()
