from src.data_ingestion import load_data, split_data
from src.data_preprocessing import scale_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.utils import save_model

DATA_PATH = "../data/raw/creditcard.csv"
MODEL_PATH = "../model/fraud_detection_model.pkl"

def main():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    model = train_model(X_train_scaled, y_train)

    evaluate_model(model, X_test_scaled, y_test)

    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()