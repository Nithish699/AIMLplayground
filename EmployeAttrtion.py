import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load dataset

df = pd.read_csv("Employee_Attrition.csv")

# Selecting significant attributes
selected_features = [
    "Age", "Years at Company", "Monthly Income", "Job Satisfaction",
    "Performance Rating", "Overtime", "Distance from Home", "Education Level",
    "Job Level", "Company Reputation"
]

target_variable = "Attrition"

# Encoding categorical variables
df_selected = df[selected_features + [target_variable]].copy()
label_encoders = {}
for col in df_selected.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_selected[col] = le.fit_transform(df_selected[col])
    label_encoders[col] = le

# Splitting dataset
X = df_selected[selected_features]
y = df_selected[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training with Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# Model evaluation
y_pred = gb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save model and encoders
model_path = "EAgradient_boosting_model.pkl"
encoder_path = "EAlabel_encoders.pkl"
joblib.dump(gb_model, model_path)
joblib.dump(label_encoders, encoder_path)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
