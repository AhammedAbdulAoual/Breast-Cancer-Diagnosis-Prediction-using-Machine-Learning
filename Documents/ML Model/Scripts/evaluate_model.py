import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv', header=None).values.ravel()


decision_tree = joblib.load('models/decision_tree_model.pkl')
knn = joblib.load('models/knn_model.pkl')


dt_predictions = decision_tree.predict(X_test)
knn_predictions = knn.predict(X_test)

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

dt_accuracy, dt_precision, dt_recall, dt_f1 = evaluate_model(y_test, dt_predictions)
knn_accuracy, knn_precision, knn_recall, knn_f1 = evaluate_model(y_test, knn_predictions)

print(f"Decision Tree - Accuracy: {dt_accuracy}, Precision: {dt_precision}, Recall: {dt_recall}, F1 Score: {dt_f1}")
print(f"KNN - Accuracy: {knn_accuracy}, Precision: {knn_precision}, Recall: {knn_recall}, F1 Score: {knn_f1}")

better_model = "Decision Tree" if dt_f1 > knn_f1 else "KNN"
print(f"The better model based on F1-score is: {better_model}")
