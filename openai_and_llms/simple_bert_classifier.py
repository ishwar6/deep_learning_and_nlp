import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class SimpleBERTClassifier(nn.Module):
    """
    A simple BERT-based classifier for text classification.
    """
    def __init__(self, num_classes):
        super(SimpleBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def preprocess_data(texts):
    """
    Tokenize input texts and create attention masks.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return encoding['input_ids'], encoding['attention_mask']

def main():
    texts = ["I love deep learning!", "Transformers are amazing!"]
    input_ids, attention_mask = preprocess_data(texts)
    model = SimpleBERTClassifier(num_classes=2)
    outputs = model(input_ids, attention_mask)
    print(outputs)

if __name__ == '__main__':
    main()