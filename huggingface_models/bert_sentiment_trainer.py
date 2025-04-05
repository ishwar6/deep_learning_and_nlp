import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis using BERT."""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    """Creates a DataLoader from a DataFrame."""
    ds = SentimentDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size)

def train_model(model, data_loader, optimizer, device):
    """Trains the model for one epoch."""
    model = model.train()
    total_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(data_loader)

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    texts = ['I love this!', 'This is bad.']
    labels = [1, 0]
    df = pd.DataFrame({'text': texts, 'label': labels})
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(df.text, df.label, test_size=0.2)
    train_data_loader = create_data_loader(pd.DataFrame({'text': train_texts, 'label': train_labels}), tokenizer, max_len=32, batch_size=2)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss = train_model(model, train_data_loader, optimizer, device)
    print(f'Training loss: {loss}')