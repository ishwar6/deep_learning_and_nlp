import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertTextClassifier(nn.Module):
    """BERT-based text classification model."""
    def __init__(self, num_classes):
        super(BertTextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

def prepare_data(texts, labels):
    """Tokenizes input texts and prepares them for model input."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels)

if __name__ == '__main__':
    texts = ['I love machine learning', 'Deep learning is fascinating', 'Natural Language Processing with BERT']
    labels = [0, 1, 1]
    input_ids, attention_mask, labels_tensor = prepare_data(texts, labels)
    model = BertTextClassifier(num_classes=2)
    output = model(input_ids, attention_mask)
    print(output)