import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class SimpleTranslator(nn.Module):
    """A simple neural machine translation model using BERT as encoder."""
    def __init__(self):
        super(SimpleTranslator, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 10000)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state)
        return logits

def preprocess_text(text, tokenizer):
    """Tokenizes input text and returns input IDs and attention mask."""
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors='pt',
        padding='max_length',
        max_length=50,
        truncation=True
    )
    return encoding['input_ids'], encoding['attention_mask']

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SimpleTranslator()
    text = "Hello, how are you?"
    input_ids, attention_mask = preprocess_text(text, tokenizer)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    print(output.shape)