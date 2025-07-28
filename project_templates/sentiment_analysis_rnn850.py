import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator
from sklearn.metrics import accuracy_score

class SentimentRNN(nn.Module):
    """A simple RNN model for sentiment analysis."""
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

def load_data():
    """Loads and preprocesses the IMDB dataset."""
    TEXT = Field(tokenize='spacy', lower=True)
    LABEL = Field(sequential=False)
    train_data, test_data = IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)
    train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=64, device='cuda')
    return train_iterator, test_iterator, len(TEXT.vocab), len(LABEL.vocab)

def train_model(model, iterator, optimizer, criterion):
    """Trains the model for one epoch."""
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = accuracy_score(batch.label.cpu(), torch.round(torch.sigmoid(predictions)).cpu())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def main():
    """Main function to execute training and evaluation."""
    train_iterator, test_iterator, vocab_size, output_size = load_data()
    model = SentimentRNN(vocab_size, 100, 256, output_size)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_iterator, optimizer, criterion)
        print(f'Epoch {epoch+1}, Loss: {train_loss:.3f}, Accuracy: {train_acc:.3f}')

if __name__ == '__main__':
    main()