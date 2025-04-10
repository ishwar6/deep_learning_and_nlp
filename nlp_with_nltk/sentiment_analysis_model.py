import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SentimentAnalyzer(nn.Module):
    """
    A simple sentiment analysis model using BERT.
    """
    def __init__(self):
        super(SentimentAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the BERT model.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def preprocess_data(sentences):
    """
    Tokenizes and prepares input data for the BERT model.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(sentences, truncation=True, padding=True, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

if __name__ == '__main__':
    sentences = ["I love this product!", "This is the worst experience I've ever had."]
    input_ids, attention_mask = preprocess_data(sentences)
    model = SentimentAnalyzer()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    print(outputs)