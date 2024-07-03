import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load preprocessed data
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv', header=None).values.ravel()
y_test = pd.read_csv('data/y_test.csv', header=None).values.ravel()

# Initialize models
decision_tree = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier()

# Train models
decision_tree.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Save the models
joblib.dump(decision_tree, 'models/decision_tree_model.pkl')
joblib.dump(knn, 'models/knn_model.pkl')

print("Models trained and saved.")
