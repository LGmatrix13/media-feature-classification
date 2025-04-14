from argparse import ArgumentParser, Namespace
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

def custom_tokenizer(text: str):
    words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text.lower())
    return words

def check_baseline(test: pd.DataFrame, label_col: str):
    label_counts = test[label_col].value_counts()
    dominant_label = label_counts.idxmax()
    correct = (test[label_col] == dominant_label).sum()
    return correct / len(test)

from wordcloud import WordCloud
def make_decision_tree_wordcloud(model: DecisionTreeClassifier, feature_names: list,max_words=15):
    importances = model.feature_importances_
    word_importances = {word: importance for word, importance in zip(feature_names, importances) if importance > 0}

    top_words = dict(sorted(word_importances.items(), key=lambda item: item[1], reverse=True)[:max_words])
    width, height = 1200, 600
    wc = WordCloud(
        width=width,
        height=height,
        background_color='white',
        max_words=max_words,
        colormap='plasma'
    )
    wc.generate_from_frequencies(top_words)

    # Display the image using a figure that matches WordCloud dimensions
    plt.figure(figsize=(width / 100, height / 100), dpi=100)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)  # Remove padding around the word cloud
    plt.show()
  
def train_and_evaluate_model(X_train, X_test, y_train, y_test, description=""):
    param_grid = {
        'criterion': ['entropy', 'gini'],
        'splitter': ['best'],
        'max_depth': [5, 10, 15],
        'max_features': ['sqrt', 'log2']
    }

    model = DecisionTreeClassifier()
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    make_decision_tree_wordcloud(best_model, X_train.columns)
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\n[{description}] Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")

    plt.figure(figsize=(20, 10))
    plot_tree(
        best_model,
        feature_names=X_train.columns,
        class_names=grid_search.classes_,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(f"Decision Tree ({description})")
    plt.show()
    return acc, f1

def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, description=""):
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None, 20],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)  

    best_model = grid_search.best_estimator_
    make_decision_tree_wordcloud(best_model, X_train.columns)
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n[{description}] Random Forest Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
    return acc, f1

def train_and_evaluate_naive_bayes(X_train, X_test, y_train, y_test, description=""):
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n[{description}] Naive Bayes Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
    return acc, f1


def main(args: Namespace):
    df = pd.read_parquet('./data/transformed/book_descriptions_2.parquet')
    sample = df.sample(n=args.n)
    sample = sample[~sample['summary'].isnull()]
    sample['ascii'] = sample['summary'].apply(lambda row: row.isascii())
    sample = sample[sample['ascii'] == True]
    ids = sample['id'].values
    documents = sample['summary'].values

    vectorizer = CountVectorizer(lowercase=True, stop_words='english', tokenizer=custom_tokenizer)
    bow_matrix = vectorizer.fit_transform(documents)
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    bow_df['id'] = ids
    print(f'number of words {len(bow_df.columns)}')

    def filter(df: pd.DataFrame):
        df.drop(columns=df.columns[df.sum() < 10], inplace=True)

    filter(bow_df)
    print(f'number of words {len(bow_df.columns)}')

    metadata = pd.read_parquet('./data/transformed/book_metadata_2.parquet')
    genres = metadata[['id', 'genre']]
    df = bow_df.merge(genres, on='id')
    df = df[~df['genre'].isnull()].copy()

    X = df.drop(columns=['genre', 'id'])
    y = df['genre']

    print(f"\n[INFO] Number of classes: {y.nunique()}")
    print("[INFO] Sample class distribution:")
    print(y.value_counts().head(10))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    baseline_correct = check_baseline(df, 'genre')
    print(f"\nBaseline Accuracy (Most Frequent Genre): {baseline_correct:.2f}")

    selector = SelectKBest(score_func=chi2, k=round((len(X.columns)) / 2))
    X_new = selector.fit_transform(X_train, y_train)
    selected_features = X.columns[selector.get_support()]
    X_train = pd.DataFrame(X_new, columns=selected_features)
    X_test = X_test[selected_features]
    print(f'number of words {len(selected_features)}')

    acc, f1 = train_and_evaluate_model(X_train, X_test, y_train, y_test, description="Multi-Class Decision Tree")
    rf_acc, rf_f1 = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, description="Multi-Class Random Forest")
    nb_acc, nb_f1 = train_and_evaluate_naive_bayes(X_train, X_test, y_train, y_test, description="Multinomial Naive Bayes")

    print(f"\nFinal Scores:")
    print(f" - Decision Tree Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
    print(f" - Random Forest Accuracy: {rf_acc:.2f}, F1 Score: {rf_f1:.2f}")
    print(f" - Naive Bayes Accuracy: {nb_acc:.2f}, F1 Score: {nb_f1:.2f}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Train a multi-class movie genre classifier")
    parser.add_argument('n', type=int, help='Number of movie overviews to sample')
    args = parser.parse_args()
    main(args)
