import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class MultilingualNLPModel(nn.Module):
    """A simple multilingual NLP model using BERT for sentence embeddings."""
    def __init__(self):
        super(MultilingualNLPModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        """Forward pass to get predictions from the model."""
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        return self.fc(pooled_output)

def tokenize_sentences(sentences):
    """Tokenizes input sentences using a multilingual BERT tokenizer."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    return tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

def main():
    sentences = ["Hello, how are you?", "Bonjour, comment ça va?"]
    tokens = tokenize_sentences(sentences)
    model = MultilingualNLPModel()
    with torch.no_grad():
        predictions = model(tokens['input_ids'], tokens['attention_mask'])
    print(predictions)

if __name__ == '__main__':
    main()