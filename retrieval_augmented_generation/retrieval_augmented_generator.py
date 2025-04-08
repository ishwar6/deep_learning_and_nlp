import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class RetrievalAugmentedGenerator:
    def __init__(self):
        """Initializes the model with a tokenizer and a BERT model."""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def encode_text(self, text):
        """Encodes input text to BERT embeddings."""
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def retrieve_and_generate(self, query, context):
        """Retrieves context based on the query and generates embeddings for both."""
        query_embedding = self.encode_text(query)
        context_embedding = self.encode_text(context)
        similarity = torch.cosine_similarity(query_embedding, context_embedding)
        return similarity.item()

if __name__ == '__main__':
    rag = RetrievalAugmentedGenerator()
    query = "What is the capital of France?"
    context = "The capital of France is Paris."
    similarity_score = rag.retrieve_and_generate(query, context)
    print(f'Similarity Score: {similarity_score}')