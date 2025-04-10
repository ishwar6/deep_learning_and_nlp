import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

class TopicModel:
    def __init__(self, n_topics=5):
        """Initializes the topic model with the given number of topics."""
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = NMF(n_components=self.n_topics, random_state=0)

    def fit(self, documents):
        """Fits the NMF model on the provided documents."""
        tfidf = self.vectorizer.fit_transform(documents)
        self.model.fit(tfidf)

    def display_topics(self, n_words=10):
        """Displays the top words for each topic."""
        feature_names = self.vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.model.components_):
            print(f'Topic {topic_idx + 1}:')
            print(' '.join([feature_names[i] for i in topic.argsort()[-n_words:]]))

if __name__ == '__main__':
    newsgroups = fetch_20newsgroups(subset='all')['data']
    model = TopicModel(n_topics=5)
    model.fit(newsgroups)
    model.display_topics(n_words=5)