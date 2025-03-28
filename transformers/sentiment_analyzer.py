import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SentimentAnalyzer(nn.Module):
    """A sentiment analysis model using BERT."""
    def __init__(self):
        super(SentimentAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return self.sigmoid(logits)

def preprocess_data(sentences):
    """Tokenize and prepare sentences for the model."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    return inputs['input_ids'], inputs['attention_mask']

def main():
    """Main function to run the sentiment analysis model."""
    sentences = ["I love this!", "This is terrible."]
    input_ids, attention_mask = preprocess_data(sentences)
    model = SentimentAnalyzer()
    with torch.no_grad():
        predictions = model(input_ids, attention_mask)
    print(predictions)

if __name__ == '__main__':
    main()