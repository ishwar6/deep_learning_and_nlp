import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertClassifier(nn.Module):
    """
    A simple BERT-based classifier for sentiment analysis.
    """
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)

def prepare_data(sentences):
    """
    Tokenizes and encodes input sentences for BERT.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    return encoding['input_ids'], encoding['attention_mask']

def main():
    sentences = ["I love programming!", "Deep learning is fascinating.", "BERT models are powerful."]
    input_ids, attention_mask = prepare_data(sentences)
    model = BertClassifier(num_classes=3)
    outputs = model(input_ids, attention_mask)
    print(outputs)

if __name__ == '__main__':
    main()