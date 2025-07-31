import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ZeroShotClassifier:
    def __init__(self, model_name):
        """Initializes the ZeroShotClassifier with a pre-trained model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def classify(self, text, candidate_labels):
        """Classifies the input text into candidate labels using a zero-shot approach."""
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        logits = outputs.last_hidden_state.mean(dim=1)
        return logits.detach().numpy()

if __name__ == '__main__':
    model_name = 'facebook/bart-large-mnli'
    classifier = ZeroShotClassifier(model_name)
    text = "The new movie was thrilling and full of surprises."
    candidate_labels = ["positive", "negative", "neutral"]
    result = classifier.classify(text, candidate_labels)
    print(f'Logits: {result}')