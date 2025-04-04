import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class RetrievalAugmentedGenerator:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def encode_text(self, text):
        """Encode input text using BERT tokenizer and model."""
        tokens = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1)

    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings."""
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(embedding1, embedding2)

if __name__ == '__main__':
    rag = RetrievalAugmentedGenerator()
    text1 = 'The cat sits on the mat.'
    text2 = 'A cat is on the mat.'
    embedding1 = rag.encode_text(text1)
    embedding2 = rag.encode_text(text2)
    similarity = rag.calculate_similarity(embedding1, embedding2)
    print(f'Similarity: {similarity.item()}')