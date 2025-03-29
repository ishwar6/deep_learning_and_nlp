import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class SimpleBertClassifier(nn.Module):
    """A simple binary classifier using BERT as the backbone."""
    def __init__(self):
        super(SimpleBertClassifier, self).__init__()  
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return self.sigmoid(logits)

def preprocess_data(sentences):
    """Tokenizes and encodes the input sentences."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    return inputs['input_ids'], inputs['attention_mask']

def train_model(model, data, labels, epochs=3):
    """Trains the BERT classifier on the provided data."""
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.BCELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(*data)
        loss = loss_fn(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    sentences = ["I love deep learning!", "Natural language processing is fascinating.", "This is an example sentence."]
    labels = torch.tensor([1, 1, 0])
    input_ids, attention_mask = preprocess_data(sentences)
    model = SimpleBertClassifier()
    train_model(model, (input_ids, attention_mask), labels)