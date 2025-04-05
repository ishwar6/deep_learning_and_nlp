import spacy
from transformers import pipeline

class SentimentAnalyzer:
    """A class to analyze sentiment from text using SpaCy and Hugging Face Transformers."""
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment_pipeline = pipeline('sentiment-analysis')

    def analyze_sentiment(self, text):
        """Analyzes the sentiment of the given text."""
        doc = self.nlp(text)
        sentiment = self.sentiment_pipeline(text)[0]
        return {
            'text': text,
            'sentiment': sentiment['label'],
            'score': sentiment['score'],
            'tokens': [token.text for token in doc]
        }

if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    sample_text = "I love using SpaCy for natural language processing!"
    result = analyzer.analyze_sentiment(sample_text)
    print(result)