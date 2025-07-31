import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertTextClassifier(nn.Module):
    """
    A simple BERT-based text classifier.
    """
    def __init__(self, num_classes):
        super(BertTextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the model.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state[:, 0, :])
        return logits

def preprocess_data(texts):
    """
    Tokenizes and encodes the input texts.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return encoding['input_ids'], encoding['attention_mask']

if __name__ == '__main__':
    texts = ['I love programming.', 'Transformers are amazing!']
    input_ids, attention_mask = preprocess_data(texts)
    model = BertTextClassifier(num_classes=2)
    outputs = model(input_ids, attention_mask)
    print(outputs)