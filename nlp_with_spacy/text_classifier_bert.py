import spacy
from spacy.tokens import Doc
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        """
        Initializes the TextClassifier with a BERT model.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess(self, texts):
        """
        Tokenizes and encodes the input texts.
        """
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def predict(self, texts):
        """
        Predicts the class labels for the given texts.
        """
        inputs = self.preprocess(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        return predictions.cpu().numpy()

# Sample usage
if __name__ == '__main__':
    classifier = TextClassifier()
    sample_texts = ['I love programming!', 'This is a bad experience.']
    predictions = classifier.predict(sample_texts)
    print('Predictions:', predictions)