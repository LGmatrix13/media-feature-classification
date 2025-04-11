from argparse import ArgumentParser, Namespace
import re
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud

def custom_tokenizer(text: str):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())  # Regex to match only words with alphabetic characters
    return words

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from better_profanity import profanity

def make_decision_tree_wordcloud(model: DecisionTreeClassifier, feature_names: list, max_words=100):
    profanity.load_censor_words()
    importances = model.feature_importances_
    word_importances = {}

    for word, importance in zip(feature_names, importances):
        if importance > 0:
            censored = profanity.censor(word)
            word_importances[censored] = importance

    # Limit to top N most important words
    top_words = dict(sorted(word_importances.items(), key=lambda item: item[1], reverse=True)[:max_words])

    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=max_words
    )
    wc.generate_from_frequencies(top_words)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Important Decision Tree Words (Censored)")
    plt.show()

def check_baseline(test: pd.DataFrame):
    # No longer filter by specific tags
    label_counts = test['TAG_LABEL'].value_counts()
    dominant_label = label_counts.idxmax()
    correct = (test['TAG_LABEL'] == dominant_label).sum()  # Use dominant tag for baseline
    return correct / len(test)

def train_and_evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, description: str = ""):
    param_grid = {
        'criterion': ['entropy', 'gini', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': list(range(1, 11)),
        'max_features': ['sqrt', 'log2']
    }

    model = DecisionTreeClassifier()
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    make_decision_tree_wordcloud(best_model, X_train.columns)
    y_pred = best_model.predict(X_test)

    # Calculate accuracy and F1 score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # You can adjust the average method as needed

    print(f"\n[{description}] Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
    plt.figure(figsize=(20, 10))  # Increase size
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

def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, description: str = ""):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
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
    # Load and prepare the data
    df = pd.read_parquet('./data/transformed/music_lyrics.parquet')
    sample = df.sample(n=args.n)
    del df
    sample = sample[~sample['lyrics'].isnull()]
    sample['ascii'] = sample['lyrics'].apply(lambda row: row.isascii())
    sample = sample[sample['ascii'] == True]
    ids = sample['id'].values  # Extract IDs
    documents = sample['lyrics'].values
    vectorizer = CountVectorizer(lowercase=True, stop_words='english', tokenizer=custom_tokenizer)
    bow_matrix = vectorizer.fit_transform(documents)

    # Load the tags dataframe
    metadata = pd.read_parquet('./data/transformed/music_metadata.parquet')
    tags = metadata[['id', 'tag']]
    tags.columns = ['id', 'TAG_LABEL']
    del metadata

    # Create Bag of Words DataFrame
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    bow_df['id'] = ids  # Assign IDs directly
    print(f"original {len(bow_df.columns)}")
    # Merge Bow and Tags
    def filter(df: pd.DataFrame):
        df.drop(columns=df.columns[df.sum() < 30], inplace=True)
    
    filter(bow_df)

    df = bow_df.merge(tags, on="id")
    X = df.drop(columns=['TAG_LABEL', 'id'])  # Features (word counts)
    y = df['TAG_LABEL']  # Target (genre)
    print(f"filtered {len(bow_df.columns)}")

    # Train/test split happens only once at the beginning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate baseline before feature selection
    baseline_correct = check_baseline(df)
    print(f"Baseline Correct Predictions (Before Feature Selection): {baseline_correct}")

    selector = SelectKBest(score_func=chi2, k=round((len(X.columns) - 2) / 2))
    X_new = selector.fit_transform(X_train, y_train)
    selected_features = X.columns[selector.get_support()]
    X_train = pd.DataFrame(X_new, columns=selected_features)
    X_test = X_test[selected_features]  # Apply feature sselection to test set as well
    print(f"feature selected {len(selected_features)}")

    # Evaluate baseline after feature selection
    baseline_correct = check_baseline(df)
    print(f"Baseline Correct Predictions (After Feature Selection): {baseline_correct}")

    acc, f1 = train_and_evaluate_model(X_train, X_test, y_train, y_test, description="After Feature Selection")
    rf_acc, rf_f1 = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, description="After Feature Selection")
    nb_acc, nb_f1 = train_and_evaluate_naive_bayes(X_train, X_test, y_train, y_test, description="Multinomial Naive Bayes")
    print(f"\nFinal Scores:")
    print(f" - Decision Tree Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
    print(f" - Random Forest Accuracy: {rf_acc:.2f}, F1 Score: {rf_f1:.2f}")
    print(f" - Naive Bayes Accuracy: {nb_acc:.2f}, F1 Score: {nb_f1:.2f}")

if __name__ == "__main__":
    parser = ArgumentParser(description=(
        "The number of observations to train on"
    ))
    # Removed the `first_tag_pred` and `second_tag_pred` options
    parser.add_argument('n', type=int)    
    args = parser.parse_args()
    main(args)
