import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class TopicModel(nn.Module):
    """
    A simple topic modeling class using BERT embeddings.
    """
    def __init__(self):
        super(TopicModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 10)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the model.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.last_hidden_state[:, 0, :])

def preprocess_data(texts):
    """
    Tokenizes and encodes the input texts.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

if __name__ == '__main__':
    sample_texts = ["Deep learning for topic modeling.", "Natural language processing is fascinating.", "BERT is a powerful model."]
    inputs, masks = preprocess_data(sample_texts)
    model = TopicModel()
    outputs = model(inputs, masks)
    print(outputs)