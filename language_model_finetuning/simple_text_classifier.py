import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class SimpleTextClassifier(nn.Module):
    """A simple text classifier using a pre-trained GPT-2 model."""
    def __init__(self, num_classes):
        super(SimpleTextClassifier, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.fc = nn.Linear(self.model.config.n_embd, num_classes)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state[:, 0, :])
        return logits

def tokenize_and_encode(texts):
    """Tokenizes and encodes a list of texts into input IDs."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

def main():
    """Main function to simulate training and inference."""
    texts = ['This is a positive example.', 'This is a negative example.']
    labels = torch.tensor([1, 0])
    inputs = tokenize_and_encode(texts)
    model = SimpleTextClassifier(num_classes=2)
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    print(outputs)

if __name__ == '__main__':
    main()