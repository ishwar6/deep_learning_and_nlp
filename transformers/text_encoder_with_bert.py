import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class TextEncoder:
    """Encodes text using BERT model."""
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def encode(self, text):
        """Encodes input text to embeddings."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

if __name__ == '__main__':
    encoder = TextEncoder()
    sample_text = "Transformers are great for NLP tasks."
    embeddings = encoder.encode(sample_text)
    print(f'Embeddings shape: {embeddings.shape}')