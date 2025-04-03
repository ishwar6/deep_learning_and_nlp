import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class TopicModel(nn.Module):
    """
    A simple topic modeling neural network using BERT.
    """
    def __init__(self):
        super(TopicModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 10)  # Assuming 10 topics

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.pooler_output)

def tokenize_and_encode(texts):
    """
    Tokenizes and encodes a list of texts using BERT tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

def main():
    texts = ["Deep learning is transforming education.", "Natural language processing enables machines to understand text."]
    encoded_inputs = tokenize_and_encode(texts)
    model = TopicModel()
    outputs = model(encoded_inputs['input_ids'], encoded_inputs['attention_mask'])
    print(outputs)

if __name__ == '__main__':
    main()