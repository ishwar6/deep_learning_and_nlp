import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SentimentAnalyzer(nn.Module):
    """Sentiment analysis model based on BERT architecture."""
    def __init__(self):
        super(SentimentAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return self.softmax(logits)

def preprocess_data(sentences):
    """Tokenizes and encodes sentences for BERT input."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

def main():
    """Main function to run sentiment analysis on sample data."""
    model = SentimentAnalyzer()
    model.eval()
    sample_sentences = ["I love this product!", "This is the worst experience ever."]
    inputs = preprocess_data(sample_sentences)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    print(outputs)

if __name__ == '__main__':
    main()