import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json

df = pd.read_csv('WineQT.csv')

X = df.drop(['quality', 'Id'], axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

id3_model = DecisionTreeClassifier(criterion='entropy')
id3_model.fit(X_train, y_train)
id3_pred = id3_model.predict(X_test)
id3_accuracy = accuracy_score(y_test, id3_pred) * 100
id3_f1 = f1_score(y_test, id3_pred, average='weighted')

if id3_f1:
    model = id3_model
    model_name = "ID3 (Decision Tree)"

joblib.dump(model, 'api/model.pkl')
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'api/feature_names.pkl')

performance_data = {
    "id3": {
        "accuracy": round(id3_accuracy, 4),
        "f1_score": round(id3_f1, 4)
    },
    "model_name": model_name
}

with open('api/model_performance.json', 'w') as f:
    json.dump(performance_data, f, indent=4)

print("\nModel terbaik, nama fitur, dan data kinerja telah disimpan.")