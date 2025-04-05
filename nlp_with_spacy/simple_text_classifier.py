import spacy
from spacy.training import Example
from spacy.util import minibatch
import random

class SimpleTextClassifier:
    def __init__(self, model='en_core_web_sm'):
        self.nlp = spacy.load(model)
        self.text_cat = self.nlp.create_pipe('textcat', config={'model': 'simple_cnn'})
        self.nlp.add_pipe(self.text_cat, last=True)
        self.text_cat.add_label('POSITIVE')
        self.text_cat.add_label('NEGATIVE')

    def train(self, train_data, n_iter=10):
        self.nlp.begin_training()
        for i in range(n_iter):
            random.shuffle(train_data)
            losses = {} 
            for batch in minibatch(train_data, size=8):
                for text, annotations in batch:
                    example = Example.from_dict(self.nlp.make_doc(text), annotations)
                    self.nlp.update([example], drop=0.5, losses=losses)
            print(f'Iteration {i + 1}, Losses: {losses}')

    def predict(self, text):
        doc = self.nlp(text)
        return doc.cats

if __name__ == '__main__':
    train_examples = [
        ('I love this product!', {'cats': {'POSITIVE': 1, 'NEGATIVE': 0}}),
        ('This is the worst experience ever.', {'cats': {'POSITIVE': 0, 'NEGATIVE': 1}}),
        ('Absolutely fantastic!', {'cats': {'POSITIVE': 1, 'NEGATIVE': 0}}),
        ('Not my taste.', {'cats': {'POSITIVE': 0, 'NEGATIVE': 1}})
    ]
    classifier = SimpleTextClassifier()
    classifier.train(train_examples, n_iter=5)
    prediction = classifier.predict('I would recommend this to everyone!')
    print(f'Prediction: {prediction}')