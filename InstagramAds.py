import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset

df = pd.read_excel("Dataset_master.xlsx", sheet_name="Instagram Ads SVM")

# Define features and target
X = df.drop(columns=["Purchased"])
y = df["Purchased"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print("-"*50)

evaluate_model(svm_model, "SVM")
evaluate_model(dt_model, "Decision Tree")
evaluate_model(rf_model, "Random Forest")

# Save models
joblib.dump(svm_model, "Instasvm_model.pkl")
joblib.dump(dt_model, "Instadt_model.pkl")
joblib.dump(rf_model, "Instarf_model.pkl")
joblib.dump(scaler, "Instascaler.pkl")

