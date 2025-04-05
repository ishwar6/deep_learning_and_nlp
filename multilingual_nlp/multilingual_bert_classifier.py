import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class MultilingualBERT(nn.Module):
    """A simple wrapper around BERT for multilingual text classification."""
    def __init__(self, num_classes):
        super(MultilingualBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

def tokenize_input(sentences):
    """Tokenizes input sentences for BERT."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    return tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

def main():
    """Main function to demonstrate model usage."""
    sentences = ["Bonjour tout le monde", "Hello world", "Hola mundo"]
    tokenized = tokenize_input(sentences)
    model = MultilingualBERT(num_classes=3)
    output = model(tokenized['input_ids'], tokenized['attention_mask'])
    print(output)

if __name__ == '__main__':
    main()