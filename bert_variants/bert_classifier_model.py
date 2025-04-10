import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertClassifier(nn.Module):
    """A simple BERT-based classifier for binary classification."""
    def __init__(self, num_classes=2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)

def tokenize_data(sentences):
    """Tokenizes input sentences using BERT tokenizer."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

if __name__ == '__main__':
    sentences = ["I love deep learning!", "BERT is a powerful model."]
    tokens = tokenize_data(sentences)
    model = BertClassifier()
    output = model(tokens['input_ids'], tokens['attention_mask'])
    print(output)