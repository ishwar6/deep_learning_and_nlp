import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertTextClassifier(nn.Module):
    """A simple BERT-based text classification model."""
    def __init__(self, num_classes):
        super(BertTextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """Forward pass through BERT and classifier."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits

def preprocess_data(texts):
    """Tokenizes and encodes the input texts."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

if __name__ == '__main__':
    sample_texts = ["Hello, how are you?", "Transformers are great for NLP tasks."]
    input_ids, attention_mask = preprocess_data(sample_texts)
    model = BertTextClassifier(num_classes=2)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        print(outputs)