import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

file_path = "balance_data.xlsx"
df = pd.read_excel(file_path)

X = df[["Location", "Occupation", "Activity", "Employment Status", "Gender", "Incident Type", "Day vs Night", "Province"]]
y = df["Nature of injury"]

label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_regression_classifier = LogisticRegression(random_state=42, max_iter=1000)
logistic_regression_classifier.fit(X_train_scaled, y_train)

model_filename = "trained_logistic_regression_model.joblib"
dump(logistic_regression_classifier, model_filename)

y_pred_test = logistic_regression_classifier.predict(X_test_scaled)

accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Accuracy on the test set: {accuracy_test:.2f}")

print("Classification Report on the test set:")
print(classification_report(y_test, y_pred_test))

# Print accuracy for each class
class_accuracy = {}

for label in logistic_regression_classifier.classes_:
    mask = (y_test == label)
    class_accuracy[label] = accuracy_score(y_test[mask], y_pred_test[mask])

for label, acc in class_accuracy.items():
    print(f"Accuracy for class '{label}': {acc:.2f}")
