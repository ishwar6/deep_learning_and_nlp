import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class TextEmbedder:
    """Converts text to embeddings using BERT model."""
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def tokenize(self, text):
        """Tokenizes input text into BERT-compatible tokens."""
        return self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    def get_embeddings(self, text):
        """Returns the embeddings for the input text."""
        tokens = self.tokenize(text)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

if __name__ == '__main__':
    embedder = TextEmbedder()
    sample_text = "Deep learning is revolutionizing education."
    embeddings = embedder.get_embeddings(sample_text)
    print("Embeddings:", embeddings)