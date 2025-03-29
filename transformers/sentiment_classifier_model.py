import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SentimentClassifier(nn.Module):
    """
    A sentiment classification model using BERT.
    """
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through BERT and the classifier.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def preprocess_data(texts):
    """
    Tokenizes the input texts for BERT.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    return encoding['input_ids'], encoding['attention_mask']

def main():
    """
    Main function to demonstrate the model.
    """
    texts = ["I love this product!", "This is the worst experience ever."]
    input_ids, attention_mask = preprocess_data(texts)
    model = SentimentClassifier()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    predictions = torch.argmax(logits, dim=-1)
    print(predictions.tolist())

if __name__ == '__main__':
    main()