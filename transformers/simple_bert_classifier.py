import torch
from torch import nn
from transformers import BertTokenizer, BertModel

class SimpleBertClassifier(nn.Module):
    """A simple BERT-based classifier for sentiment analysis."""
    def __init__(self, num_classes:int):
        super(SimpleBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits

def preprocess_data(texts):
    """Tokenizes input texts and prepares them for BERT."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return encodings.input_ids, encodings.attention_mask

if __name__ == '__main__':
    sample_texts = ['I love programming!', 'This is a bad example.']
    input_ids, attention_mask = preprocess_data(sample_texts)
    model = SimpleBertClassifier(num_classes=2)
    outputs = model(input_ids, attention_mask)
    print(outputs.detach().numpy())