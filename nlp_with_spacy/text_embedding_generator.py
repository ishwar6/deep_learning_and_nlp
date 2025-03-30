import spacy
from spacy.tokens import Doc
import torch
from transformers import BertTokenizer, BertModel

class CustomTokenizer:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors='pt')

class TextEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.model = BertModel.from_pretrained(model_name)

    def embed(self, tokens):
        with torch.no_grad():
            outputs = self.model(**tokens)
            return outputs.last_hidden_state

    def get_embedding(self, text):
        tokens = self.tokenize(text)
        embeddings = self.embed(tokens)
        return embeddings.mean(dim=1).squeeze().numpy()

    def print_embedding(self, text):
        embedding = self.get_embedding(text)
        print(f'Embedding for \'{text}\': {embedding})

if __name__ == '__main__':
    text = 'Deep learning transforms NLP.'
    embedder = TextEmbedder()
    embedder.print_embedding(text)