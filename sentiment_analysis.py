
# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define preprocessing functions
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    words = word_tokenize(text.lower())
    filtered_words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
    return " ".join(filtered_words)

# Sample dataset (can be replaced with a more extensive dataset)
data = {
    'text': [
        "I love this product! It's amazing.",
        "Terrible experience, would not recommend.",
        "I'm not sure how I feel about this.",
        "Great service and friendly staff!",
        "This was the worst purchase I've ever made.",
        "It was okay, nothing special."
    ],
    'label': ['positive', 'negative', 'neutral', 'positive', 'negative', 'neutral']
}

# Load data into a DataFrame and preprocess text
df = pd.DataFrame(data)
df['processed_text'] = df['text'].apply(preprocess_text)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict sentiment of new text
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_vec = vectorizer.transform([processed_text])
    prediction = model.predict(text_vec)
    return prediction[0]

# Example usage
print("Predicted Sentiment:", predict_sentiment("The product is absolutely wonderful!"))
