import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

file_path = "balance_data.xlsx"
df = pd.read_excel(file_path)

# columns's name
X = df[["Location", "Occupation", "Activity", "Employment Status", "Gender", "Incident Type", "Day vs Night", "Province"]]
y = df["Nature of injury"]

label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Save the trained model
model_filename = "trained_random_forest_model.joblib"
dump(rf_classifier, model_filename)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print accuracy for each class
class_accuracy = {}

for label in rf_classifier.classes_:
    mask = (y_test == label)
    class_accuracy[label] = accuracy_score(y_test[mask], y_pred[mask])

for label, acc in class_accuracy.items():
    print(f"Accuracy for class '{label}': {acc:.2f}")
