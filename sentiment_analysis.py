import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset from the file
file_path = 'sentiment_dataset.txt'
df = pd.read_csv(file_path, delimiter=r'\s{2,}', header=None, names=['text', 'label'], engine='python')
df.dropna(inplace=True)

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize classifiers
classifiers = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(kernel='linear')
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

# Choose the best-performing model for prediction
best_model = classifiers["Logistic Regression"]

# Define a function to predict sentiment
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_vec = vectorizer.transform([processed_text])
    prediction = best_model.predict(text_vec)
    return prediction[0]

# Example usage
print("Predicted Sentiment:", predict_sentiment("The product is absolutely wonderful!"))
