from transformers import BertTokenizer, BertForSequenceClassification
import torch

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

    def preprocess(self, texts):
        """Tokenizes and encodes the input texts."
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def predict(self, texts):
        """Predicts the class for the given texts."
        inputs = self.preprocess(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        return torch.argmax(logits, dim=-1)

if __name__ == '__main__':
    classifier = TextClassifier()
    sample_texts = ['I love programming.', 'Deep learning is fascinating.']
    predictions = classifier.predict(sample_texts)
    print(predictions.tolist())