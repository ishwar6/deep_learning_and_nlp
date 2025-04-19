import spacy
from spacy.tokens import Doc
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess(self, texts):
        """Tokenizes and encodes the input texts."""
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def predict(self, texts):
        """Makes predictions on the input texts."""
        inputs = self.preprocess(texts)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.argmax(logits, dim=1)

def main():
    nlp = spacy.load('en_core_web_sm')
    classifier = TextClassifier()
    texts = ['I love using spaCy for NLP tasks!', 'Deep learning is fascinating.']
    predictions = classifier.predict(texts)
    for text, pred in zip(texts, predictions):
        print(f'Text: {text} | Prediction: {pred.item()}')

if __name__ == '__main__':
    main()