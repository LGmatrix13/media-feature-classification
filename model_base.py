import nltk
import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

class GenreClassifier:
    def __init__(self, data_path: str, metadata_path: str, paratext_col: str, genre_col: str, n: int):
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)

        self.data_path = data_path
        self.metadata_path = metadata_path
        self.paratext_col = paratext_col
        self.n = n
        self.genre_col = genre_col
        self.vectorizer = CountVectorizer(lowercase=True, tokenizer=self.custom_tokenizer)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
    
    def custom_tokenizer(self, text: str):
        words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text.lower())
        tokens = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return stemmed_tokens

    def check_baseline(self, test: pd.DataFrame, label_col: str):
        label_counts = test[label_col].value_counts()
        dominant_label = label_counts.idxmax()
        correct = (test[label_col] == dominant_label).sum()
        return correct / len(test)

    def vectorize_data(self, documents: list[str]):
        bow_matrix = self.vectorizer.fit_transform(documents)
        bow_df = pd.DataFrame(bow_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())
        return bow_df

    def filter_data(self, df: pd.DataFrame):
        """Remove columns where the total count (sum) is less than 15."""
        freq = (df.drop(columns=['id'])).sum(axis=0)
        cols_to_keep = freq[freq >= 15].index
        cols_to_keep = list(cols_to_keep) + ['id']
        df.drop(columns=[col for col in df.columns if col not in cols_to_keep], inplace=True)

    def train_and_evaluate_model(self, X_train, X_test, y_train, y_test, description=""):
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
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"\n[{description}] Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
        return acc, f1

    def train_and_evaluate_random_forest(self, X_train, X_test, y_train, y_test, description=""):
        param_grid = {
            'n_estimators': [100],
            'max_depth': [None, 20],
            'max_features': ['sqrt', 'log2'],
            'criterion': ['gini', 'entropy']
        }

        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"\n[{description}] Random Forest Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
        return acc, f1

    def train_and_evaluate_naive_bayes(self, X_train, X_test, y_train, y_test, description=""):
        model = MultinomialNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"\n[{description}] Naive Bayes Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
        return acc, f1

    def train_and_evaluate_ensemble(self, X_train, X_test, y_train, y_test, description=""):
        # Use VotingClassifier (soft voting if possible)
        clf1 = LogisticRegression(max_iter=1000, solver='liblinear')
        clf2 = MultinomialNB()
        clf3 = RandomForestClassifier(n_estimators=100, random_state=42)

        ensemble = VotingClassifier(estimators=[
            ('lr', clf1),
            ('nb', clf2),
            ('rf', clf3)
        ], voting='soft', n_jobs=-1)

        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"\n[{description}] Ensemble (Logistic+NB+RF) Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
        return acc, f1

    def train(self):
        df = pd.read_parquet(self.data_path)
        sample = df.sample(n=self.n)
        sample = sample[~sample[self.paratext_col].isnull()]
        sample['ascii'] = sample[self.paratext_col].apply(lambda row: row.isascii())
        sample = sample[sample['ascii'] == True]
        ids = sample['id'].values
        documents = sample[self.paratext_col].values

        bow_df = self.vectorize_data(documents)
        bow_df['id'] = ids
        print(f'number of words {len(bow_df.columns)}')

        self.filter_data(bow_df)
        print(f'number of words {len(bow_df.columns)}')

        metadata = pd.read_parquet(self.metadata_path)
        genres = metadata[['id', self.genre_col]]
        genres.columns = ['id', f"{self.genre_col}_GENRE"]
        df = bow_df.merge(genres, on='id')
        df = df[~df[f"{self.genre_col}_GENRE"].isnull()].copy()
        X = df.drop(columns=[f"{self.genre_col}_GENRE", 'id'])
        y = df[f"{self.genre_col}_GENRE"]

        print(f"\n[INFO] Number of classes: {y.nunique()}")
        print("[INFO] Sample class distribution:")
        print(y.value_counts().head(10))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        baseline_correct = self.check_baseline(df, f"{self.genre_col}_GENRE")
        print(f"\nBaseline Accuracy (Most Frequent Genre): {baseline_correct:.2f}")

        k = len(X.columns) if len(X.columns) > 2000 else 2000
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_new = selector.fit_transform(X_train, y_train)
        selected_features = X.columns[selector.get_support()]
        X_train = pd.DataFrame(X_new, columns=selected_features)
        X_test = X_test[selected_features]
        print(f'number of words {len(selected_features)}')

        acc, f1 = self.train_and_evaluate_model(X_train, X_test, y_train, y_test, description="Multi-Class Decision Tree")
        rf_acc, rf_f1 = self.train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, description="Multi-Class Random Forest")
        nb_acc, nb_f1 = self.train_and_evaluate_naive_bayes(X_train, X_test, y_train, y_test, description="Multinomial Naive Bayes")
        ens_acc, ens_f1 = self.train_and_evaluate_ensemble(X_train, X_test, y_train, y_test, description="Ensemble (Logistic + Naive Bayes + RF)")

        print(f"\nFinal Scores:")
        print(f" - Decision Tree Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
        print(f" - Random Forest Accuracy: {rf_acc:.2f}, F1 Score: {rf_f1:.2f}")
        print(f" - Naive Bayes Accuracy: {nb_acc:.2f}, F1 Score: {nb_f1:.2f}")
        print(f" - Ensemble Accuracy: {ens_acc:.2f}, F1 Score: {ens_f1:.2f}")
