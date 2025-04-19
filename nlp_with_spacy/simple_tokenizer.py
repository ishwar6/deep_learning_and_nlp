import spacy
from spacy.tokens import Doc
import numpy as np

class SimpleTokenizer:
    def __init__(self, model='en_core_web_sm'):
        """Initialize the tokenizer with a SpaCy model."""
        self.nlp = spacy.load(model)

    def tokenize(self, text):
        """Tokenizes input text into a list of tokens."""
        doc = self.nlp(text)
        return [token.text for token in doc]

    def encode_tokens(self, tokens):
        """Encodes tokens into numerical format using a simple mapping."""
        token_to_id = {token: idx for idx, token in enumerate(set(tokens))}
        return np.array([token_to_id[token] for token in tokens])

if __name__ == '__main__':
    text = 'SpaCy is an NLP library that makes text processing easy.'
    tokenizer = SimpleTokenizer()
    tokens = tokenizer.tokenize(text)
    encoded = tokenizer.encode_tokens(tokens)
    print('Tokens:', tokens)
    print('Encoded tokens:', encoded)