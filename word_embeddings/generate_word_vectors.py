import numpy as np
import pandas as pd

class WordEmbeddings:
    """
    A class to handle word embeddings using a specified corpus.
    """

    def __init__(self, corpus):
        """
        Initializes the WordEmbeddings with a text corpus.
        """
        self.corpus = corpus
        self.word_vectors = None

    def preprocess_data(self):
        """
        Preprocesses the corpus by tokenizing and removing stop words.
        """
        stop_words = set(['the', 'is', 'in', 'and', 'to', 'a'])
        tokens = [word for word in self.corpus.split() if word.lower() not in stop_words]
        return tokens

    def generate_embeddings(self):
        """
        Generates random word embeddings for each unique word in the corpus.
        """
        tokens = self.preprocess_data()
        unique_words = set(tokens)
        self.word_vectors = {word: np.random.rand(100) for word in unique_words}

    def save_embeddings(self, file_path):
        """
        Saves the generated word embeddings to a CSV file.
        """
        if self.word_vectors is None:
            raise ValueError('Word embeddings not generated. Call generate_embeddings first.')
        df = pd.DataFrame.from_dict(self.word_vectors, orient='index')
        df.to_csv(file_path, header=False)

corpus = 'The quick brown fox jumps over the lazy dog'
embeddings = WordEmbeddings(corpus)
embeddings.generate_embeddings()
embeddings.save_embeddings('word_embeddings.csv')
print('Word embeddings saved successfully.')