import pandas as pd
import re

def preprocess(df: pd.DataFrame):
    df = df[df['views'] > df['views'].quantile(0.6)]
    for i in range(1, 25):
        df.loc[df['year'] == i, 'year'] = int(f"200{i}") if i < 10 else int(f"20{i}")
    df = df[df['year'] > 1850]

    def split_features(s: str):
        s = re.sub(r'\\', '', s)
        words = re.findall(r'\"(.*?)\"', s)
        return words[:3] + [None] * (3 - len(words))

    df[['feature_1', 'feature_2', 'feature_3']] = df['features'].apply(split_features).apply(pd.Series)
    df = df.drop(columns=["features"])
    df['total_features'] = df.apply(lambda row: sum(pd.notna(row[col]) for col in ['feature_1', 'feature_2', 'feature_3']), axis=1)
    return df