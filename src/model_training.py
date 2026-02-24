from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        class_weight='balanced',
        max_depth=10,
        min_samples_split=5,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        class_weight='balanced',
        max_depth=10,
        min_samples_split=5,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def load_model(model_path="model/fraud_detection_model.pkl"):
    model = joblib.load(model_path)
    return model