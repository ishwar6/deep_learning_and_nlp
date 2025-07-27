import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class TopicModel(nn.Module):
    """ A simple topic model using BERT embeddings. """
    def __init__(self, num_topics):
        super(TopicModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_topics)

    def forward(self, input_ids, attention_mask):
        """ Forward pass through the model. """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        topic_logits = self.fc(pooled_output)
        return topic_logits

def preprocess_data(texts):
    """ Tokenizes input texts for the BERT model. """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

def train_model(model, data_loader, num_epochs=3):
    """ Trains the topic model on the provided data loader. """
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

texts = ['I love programming.', 'Deep learning is fascinating.', 'Natural language processing is key.']
n_labels = torch.tensor([0, 1, 2]).unsqueeze(0)
encoded_data = preprocess_data(texts)
model = TopicModel(num_topics=3)
train_model(model, [(encoded_data['input_ids'], encoded_data['attention_mask'], n_labels)])
print('Training complete.')