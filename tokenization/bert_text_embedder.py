import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class TextEmbedder:
    """
    A class to tokenize and embed text using BERT.
    """

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def tokenize(self, text):
        """
        Tokenizes input text using BERT tokenizer.
        """
        inputs = self.tokenizer(text, return_tensors='pt')
        return inputs

    def embed(self, text):
        """
        Generates embeddings for the input text.
        """
        inputs = self.tokenize(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

if __name__ == '__main__':
    sample_text = 'Deep learning is transforming the world of NLP.'
    embedder = TextEmbedder()
    embeddings = embedder.embed(sample_text)
    print('Generated embeddings:', embeddings)
    print('Shape of embeddings:', embeddings.shape)