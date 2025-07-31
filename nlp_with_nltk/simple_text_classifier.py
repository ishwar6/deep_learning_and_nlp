import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np

nltk.download('punkt')

def preprocess_text(text):
    """Tokenizes input text into words."""
    return word_tokenize(text.lower())

class TextClassifier:
    """A simple text classifier using Naive Bayes."""
    def __init__(self):
        self.model = MultinomialNB()

    def fit(self, X, y):
        """Fits the model to the training data."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predicts the classes for the input data."""
        return self.model.predict(X)

if __name__ == '__main__':
    documents = [
        "I love programming in Python!",
        "Natural Language Processing is fascinating.",
        "Python is great for data science.",
        "I enjoy learning new programming languages.",
        "Deep learning models can be complex."
    ]
    labels = [
        "positive",
        "positive",
        "positive",
        "positive",
        "neutral"
    ]
    processed_docs = [preprocess_text(doc) for doc in documents]
    vectorized_docs = np.array([len(doc) for doc in processed_docs]).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(vectorized_docs, labels, test_size=0.2, random_state=42)
    classifier = TextClassifier()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy:.2f}')