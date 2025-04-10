import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel

class TopicModeling:
    def __init__(self, model_name='bert-base-uncased'):
        """Initialize the topic modeling class with BERT tokenizer and model."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def encode_texts(self, texts):
        """Encode a list of texts into BERT embeddings."""
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()  # Use [CLS] token embeddings

    def find_topics(self, texts, num_topics=3):
        """Cluster texts into topics using k-means on embeddings."""
        from sklearn.cluster import KMeans
        embeddings = self.encode_texts(texts)
        kmeans = KMeans(n_clusters=num_topics)
        kmeans.fit(embeddings)
        return kmeans.labels_

if __name__ == '__main__':
    sample_texts = [
        'Deep learning is a subset of machine learning.',
        'Natural language processing allows computers to understand human language.',
        'Topic modeling helps in discovering abstract topics from documents.',
        'Neural networks are the backbone of deep learning.',
        'Clustering algorithms are used for grouping similar items.'
    ]
    topic_model = TopicModeling()
    topics = topic_model.find_topics(sample_texts, num_topics=2)
    print('Identified topics:', topics)