import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertVariantClassifier(nn.Module):
    """
    A simple classifier using BERT as a base model.
    """
    def __init__(self, num_classes):
        super(BertVariantClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def prepare_data(texts):
    """
    Prepares input data for the BERT model.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return inputs['input_ids'], inputs['attention_mask']

if __name__ == '__main__':
    texts = ["Hello, world!", "BERT models are great for NLP tasks."]
    input_ids, attention_mask = prepare_data(texts)
    model = BertVariantClassifier(num_classes=2)
    outputs = model(input_ids, attention_mask)
    print(outputs)