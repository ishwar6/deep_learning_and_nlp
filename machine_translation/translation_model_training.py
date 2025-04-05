import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator, TabularDataset
from transformers import BertTokenizer, BertModel

class TranslationModel(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        outputs = self.bert(text)[0]
        return self.fc(outputs[:, 0, :])

def load_data(file_path):
    source_field = Field(tokenize='spacy', lower=True)
    target_field = Field(tokenize='spacy', lower=True)
    fields = {'source': ('src', source_field), 'target': ('trg', target_field)}
    dataset = TabularDataset(path=file_path, format='csv', fields=fields)
    return dataset, source_field, target_field

def train_model(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def main():
    dataset, source_field, target_field = load_data('translation_data.csv')
    source_field.build_vocab(dataset)
    target_field.build_vocab(dataset)
    train_iterator, valid_iterator = BucketIterator.splits((dataset, dataset), batch_size=32)
    model = TranslationModel(hidden_dim=768, output_dim=len(target_field.vocab)).to('cuda')
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    train_loss = train_model(model, train_iterator, optimizer, criterion)
    print(f'Training Loss: {train_loss:.3f}')

if __name__ == '__main__':
    main()