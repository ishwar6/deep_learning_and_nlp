import spacy
from transformers import pipeline

class SentimentAnalyzer:
    """
    A class to analyze sentiment using Hugging Face Transformers and SpaCy.
    """

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment_pipeline = pipeline('sentiment-analysis')

    def analyze_text(self, text):
        """
        Analyzes the sentiment of the given text.
        
        Parameters:
        text (str): The text to analyze.
        
        Returns:
        dict: The sentiment analysis result.
        """
        doc = self.nlp(text)
        sentiment_result = self.sentiment_pipeline(doc.text)
        return sentiment_result[0]

if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    sample_text = "I love using SpaCy for Natural Language Processing tasks!"
    result = analyzer.analyze_text(sample_text)
    print(f'Sentiment: {result['label']}, Score: {result['score']:.4f}')