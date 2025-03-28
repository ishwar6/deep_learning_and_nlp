import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator
from sklearn.metrics import accuracy_score

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        final_output = self.fc(lstm_out[-1])
        return self.sigmoid(final_output)

def load_data():
    text_field = Field(tokenize='spacy', lower=True)
    label_field = Field(sequential=False, use_vocab=False)
    train_data, test_data = IMDB.splits(text_field, label_field)
    text_field.build_vocab(train_data)
    return train_data, test_data, len(text_field.vocab)

def train_model(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze()
        loss = criterion(predictions, batch.label.float())
        acc = accuracy_score(batch.label.cpu(), (predictions > 0.5).cpu())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def main():
    train_data, test_data, vocab_size = load_data()
    train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=64)
    model = SentimentModel(vocab_size, 100, 256)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    train_loss, train_acc = train_model(model, train_iterator, optimizer, criterion)
    print(f'Training Loss: {train_loss:.3f}, Training Accuracy: {train_acc:.3f}')

if __name__ == '__main__':
    main()