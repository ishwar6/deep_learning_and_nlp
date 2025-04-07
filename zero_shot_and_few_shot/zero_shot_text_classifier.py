import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class ZeroShotTextClassifier:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def predict(self, text, candidates):
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        logits = outputs.logits[-1, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        best_candidate_idx = torch.argmax(probabilities).item()
        return candidates[best_candidate_idx], probabilities[best_candidate_idx].item()

if __name__ == '__main__':
    classifier = ZeroShotTextClassifier()
    text = 'This is a beautiful day.'
    candidates = ['Positive', 'Negative', 'Neutral']
    prediction, confidence = classifier.predict(text, candidates)
    print(f'Prediction: {prediction}, Confidence: {confidence:.4f}')