import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SentimentAnalyzer(nn.Module):
    """
    A sentiment analysis model based on BERT architecture.
    """
    def __init__(self):
        super(SentimentAnalyzer, self).__init__()
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
    Preprocess sentences for BERT tokenization.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    return inputs['input_ids'], inputs['attention_mask']

def main():
    """
    Main function to run sentiment analysis on sample sentences.
    """
    sample_sentences = ["I love this product!", "This is the worst service ever."]
    input_ids, attention_mask = preprocess_data(sample_sentences)
    model = SentimentAnalyzer()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1)
    print("Predictions:", predictions.numpy())

if __name__ == '__main__':
    main()