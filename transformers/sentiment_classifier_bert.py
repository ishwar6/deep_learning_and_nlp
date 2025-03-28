import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SentimentClassifier(nn.Module):
    """
    A simple sentiment classifier based on BERT.
    """
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def preprocess_data(sentences):
    """
    Tokenizes and prepares input data for BERT.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    return inputs['input_ids'], inputs['attention_mask']

def main():
    """
    Main function to execute sentiment classification.
    """
    sentences = ['I love this product!', 'This is the worst thing ever.']
    input_ids, attention_mask = preprocess_data(sentences)
    model = SentimentClassifier()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    print('Logits:', logits)

if __name__ == '__main__':
    main()