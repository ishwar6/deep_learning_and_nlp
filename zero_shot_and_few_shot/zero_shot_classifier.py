import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class ZeroShotClassifier(nn.Module):
    """A model for zero-shot text classification using BERT."""
    def __init__(self):
        super(ZeroShotClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits

def classify_text(text, candidate_labels):
    """Classifies the input text into candidate labels using zero-shot learning."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = ZeroShotClassifier()
    model.eval()
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
    probabilities = torch.sigmoid(logits).squeeze().numpy()
    return {label: probabilities[i] for i, label in enumerate(candidate_labels)}

if __name__ == '__main__':
    text = "This is a great movie!"
    candidate_labels = ["positive", "negative"]
    results = classify_text(text, candidate_labels)
    print(results)