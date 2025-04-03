import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class MultilingualNLPModel(nn.Module):
    def __init__(self):
        super(MultilingualNLPModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def preprocess_text(text, tokenizer):
    return tokenizer(text, return_tensors='pt', padding=True, truncation=True)

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = MultilingualNLPModel()
    model.eval()
    sample_text = "Bonjour tout le monde"
    inputs = preprocess_text(sample_text, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
    print(outputs)

if __name__ == '__main__':
    main()