import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ZeroShotTextClassifier:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    def classify(self, text, labels):
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(input_ids)
        logits = outputs.logits[:, -1, :]
        probabilities = nn.functional.softmax(logits, dim=-1)
        label_ids = self.tokenizer.encode(labels)
        return {label: probabilities[0, id].item() for id in label_ids}

if __name__ == '__main__':
    classifier = ZeroShotTextClassifier()
    text = 'I love programming in Python.'
    labels = ['positive', 'negative', 'neutral']
    results = classifier.classify(text, labels)
    print('Classification Results:', results)