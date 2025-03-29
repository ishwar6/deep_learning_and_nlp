import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import AG_NEWS
from torchtext.data import Field, BucketIterator
from sklearn.metrics import accuracy_score

class TextClassifier(nn.Module):
    """
    A simple feedforward neural network for text classification.
    """
    def __init__(self, vocab_size, embed_size, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded_mean = embedded.mean(dim=1)
        return self.fc(embedded_mean)

def train(model, iterator, optimizer, criterion):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        text, labels = batch.text, batch.label
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(iterator)

def evaluate(model, iterator):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch.text, batch.label
            output = model(text)
            preds = output.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, predictions)

def main():
    TEXT = Field(tokenize='spacy', lower=True)
    LABEL = Field(dtype=torch.long)
    train_data, test_data = AG_NEWS.splits(TEXT, LABEL)
    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)
    train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=64)

    model = TextClassifier(len(TEXT.vocab), 100, len(LABEL.vocab))
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        train_loss = train(model, train_iterator, optimizer, criterion)
        accuracy = evaluate(model, test_iterator)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Test Accuracy: {accuracy:.3f}')

if __name__ == '__main__':
    main()