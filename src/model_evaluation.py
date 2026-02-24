from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

FINAL_THRESHOLD = 0.3

def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= FINAL_THRESHOLD).astype(int)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nROC-AUC:", roc_auc_score(y_test, y_proba))