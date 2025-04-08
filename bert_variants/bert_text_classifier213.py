import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertTextClassifier(nn.Module):
    """A simple BERT-based text classifier."""
    def __init__(self, num_classes):
        super(BertTextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """Forward pass for the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits

def preprocess_data(texts):
    """Tokenizes input texts and creates attention masks."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    return encoding['input_ids'], encoding['attention_mask']

def main():
    """Main execution function to demonstrate the model with mock data."""
    texts = ['Hello, how are you?', 'BERT variants are fascinating.']
    input_ids, attention_mask = preprocess_data(texts)
    model = BertTextClassifier(num_classes=2)
    logits = model(input_ids, attention_mask)
    print('Model output logits:', logits)

if __name__ == '__main__':
    main()