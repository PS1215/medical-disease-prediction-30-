import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Dataset
df = pd.read_csv("symptom_disease_dataset_500.csv")

# Step 2: Separate Features and Labels
# Last column is assumed to be the disease
label_column = "disease"
X = df.drop(columns=[label_column])
y = df[label_column]

# Step 3: Encode the disease labels (text → numbers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Step 5: Build the ML model
model = RandomForestClassifier(n_estimators=200, random_state=42)

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Step 9: Save the model and preprocessing files
os.makedirs("model_artifacts", exist_ok=True)

with open("model_artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model_artifacts/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("model_artifacts/feature_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Model and files saved inside 'model_artifacts' folder")
