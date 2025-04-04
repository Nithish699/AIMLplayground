import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords if not available
nltk.download('stopwords')

# Load Data
df = pd.read_excel("zomato.xlsx", sheet_name="Sheet1")

# Data Preprocessing Function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(words)

# Apply cleaning function
df['cleaned_review'] = df['Review'].astype(str).apply(clean_text)

# Generate WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df['cleaned_review']))
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud of Zomato Reviews")
#plt.savefig("zomato_wordcloud.png")  # Save the wordcloud image
plt.show()

# Convert text to numerical representation using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['Liked']  # Target Variable (1 for Positive, 0 for Negative)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# Train, Evaluate and Save Models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print(f"\n🔹 {name} Results:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    
    # Save each model
    joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")

# Save TF-IDF Vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# --------------- USER INPUT SECTION ---------------
def predict_review():
    user_review = input("\nEnter a Zomato review: ")
    cleaned_review = clean_text(user_review)
    transformed_review = vectorizer.transform([cleaned_review])

    print("\nPredictions:")
    for name, model in models.items():
        prediction = model.predict(transformed_review)[0]
        sentiment = "Liked (1)" if prediction == 1 else "Not Liked (0)"
        print(f"{name}: {sentiment}")

# Call the function to take user input
predict_review()

