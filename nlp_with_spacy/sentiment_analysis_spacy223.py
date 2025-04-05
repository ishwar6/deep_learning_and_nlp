import spacy
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analysis pipeline."""
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment_pipeline = pipeline('sentiment-analysis')

    def analyze_text(self, text):
        """Analyze the sentiment of the provided text."""
        doc = self.nlp(text)
        sentiment_result = self.sentiment_pipeline(text)
        return doc, sentiment_result

    def display_analysis(self, text):
        """Display the analysis results for the provided text."""
        doc, sentiment = self.analyze_text(text)
        print(f'Text: {text}')
        print(f'Entities: {[ent.text for ent in doc.ents]}')
        print(f'Sentiment: {sentiment[0]['label']} with score {sentiment[0]['score']:.4f}')

if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    sample_text = "I love using Spacy for NLP tasks!"
    analyzer.display_analysis(sample_text)