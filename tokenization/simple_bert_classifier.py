import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SimpleBertClassifier(nn.Module):
    """
    A simple BERT classifier for sentiment analysis.
    """
    def __init__(self):
        super(SimpleBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the BERT classifier.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def tokenize_sentences(sentences):
    """
    Tokenizes input sentences using BERT tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    return tokens['input_ids'], tokens['attention_mask']

if __name__ == '__main__':
    example_sentences = ["I love deep learning!", "Tokenization is essential for NLP."]
    input_ids, attention_mask = tokenize_sentences(example_sentences)
    model = SimpleBertClassifier()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    print(outputs)