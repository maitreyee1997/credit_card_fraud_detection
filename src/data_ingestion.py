import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(path):
    df = pd.read_csv(path)
    return df

def split_data(df, test_size=0.2):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test