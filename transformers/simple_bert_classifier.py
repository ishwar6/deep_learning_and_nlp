import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SimpleBERTClassifier(nn.Module):
    """A simple classifier using BERT for binary classification."""
    def __init__(self):
        super(SimpleBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        return self.sigmoid(self.fc(pooled_output))

def preprocess_data(sentences):
    """Tokenizes and encodes sentences using BERT tokenizer."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

if __name__ == '__main__':
    sentences = ["I love programming in Python!", "Deep learning is fascinating."]
    input_ids, attention_mask = preprocess_data(sentences)
    model = SimpleBERTClassifier()
    outputs = model(input_ids, attention_mask)
    print(outputs.detach().numpy())