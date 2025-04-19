import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class TextTokenizer:
    def __init__(self, model_name):
        """Initializes the tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def tokenize(self, text):
        """Tokenizes input text and returns token ids and attention mask."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return inputs['input_ids'], inputs['attention_mask']

    def get_embeddings(self, text):
        """Generates embeddings for the input text."""
        input_ids, attention_mask = self.tokenize(text)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state.mean(dim=1).numpy()

if __name__ == '__main__':
    model_name = 'bert-base-uncased'
    tokenizer = TextTokenizer(model_name)
    sample_text = 'Deep learning is a subfield of machine learning.'
    embeddings = tokenizer.get_embeddings(sample_text)
    print('Text Embeddings:', embeddings)