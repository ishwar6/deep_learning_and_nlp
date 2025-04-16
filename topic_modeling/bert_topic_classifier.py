import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class TopicModel(nn.Module):
    def __init__(self, num_topics, pretrained_model='bert-base-uncased'):
        super(TopicModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_topics)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def preprocess_data(texts, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

def train_model(model, data_loader, epochs=3, lr=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for inputs, masks, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

texts = ['Deep learning for topic modeling', 'Natural language processing with transformers', 'Machine learning applications in education']
input_ids, attention_masks = preprocess_data(texts)
labels = torch.tensor([0, 1, 2])

dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
loader = torch.utils.data.DataLoader(dataset, batch_size=2)

model = TopicModel(num_topics=3)
train_model(model, loader)