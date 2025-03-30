import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertClassifier(nn.Module):
    """
    A simple BERT-based classifier for binary classification tasks.
    """
    def __init__(self, num_classes=2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

def preprocess_data(sentences):
    """
    Tokenizes and encodes sentences for BERT input.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    return inputs['input_ids'], inputs['attention_mask']

def main():
    sentences = ["I love programming in Python!", "Deep learning is fascinating."]
    input_ids, attention_mask = preprocess_data(sentences)
    model = BertClassifier()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    print(logits)

if __name__ == '__main__':
    main()