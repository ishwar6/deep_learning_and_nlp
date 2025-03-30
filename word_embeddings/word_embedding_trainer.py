import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import GloVe

class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input):
        return self.embeddings(input)

def load_data(file_path):
    text_field = Field(tokenize='spacy', lower=True)
    fields = {'text': ('text', text_field)}
    dataset = TabularDataset(path=file_path, format='csv', fields=fields)
    return dataset, text_field

def train_model(dataset, text_field, embedding_dim=100, epochs=5):
    text_field.build_vocab(dataset, vectors=GloVe(name='6B', dim=embedding_dim))
    model = WordEmbeddingModel(len(text_field.vocab), embedding_dim)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch in BucketIterator(dataset, batch_size=32):
            optimizer.zero_grad()
            text_indices = batch.text[0]
            output = model(text_indices)
            loss = criterion(output.view(-1, len(text_field.vocab)), text_indices.view(-1))
            loss.backward()
            optimizer.step()
    return model

if __name__ == '__main__':
    dataset, text_field = load_data('sample_data.csv')
    model = train_model(dataset, text_field)
    print('Training complete. Model summary:')
    print(model)