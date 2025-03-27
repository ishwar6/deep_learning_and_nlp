import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def load_data():
    """Loads and splits the 20 Newsgroups dataset into training and test sets."""
    newsgroups = fetch_20newsgroups(subset='all')
    X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def vectorize_data(X_train, X_test):
    """Converts text data into TF-IDF feature vectors."""
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec


def train_model(X_train_vec, y_train):
    """Trains a Multinomial Naive Bayes model on the TF-IDF vectors."""
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model


def evaluate_model(model, X_test_vec, y_test):
    """Evaluates the trained model and prints the accuracy score."""
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')


def main():
    """Main function to execute the workflow of loading data, vectorizing, training, and evaluating the model."""
    X_train, X_test, y_train, y_test = load_data()
    X_train_vec, X_test_vec = vectorize_data(X_train, X_test)
    model = train_model(X_train_vec, y_train)
    evaluate_model(model, X_test_vec, y_test)


if __name__ == '__main__':
    main()