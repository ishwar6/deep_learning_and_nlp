import spacy
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        """Initializes the sentiment analysis pipeline using HuggingFace transformers."""
        self.nlp = pipeline('sentiment-analysis')

    def analyze(self, text):
        """Analyzes the sentiment of the provided text."""
        return self.nlp(text)

def main():
    """Main function to execute sentiment analysis on sample texts."""
    analyzer = SentimentAnalyzer()
    sample_texts = [
        "I love using Spacy for NLP tasks!",
        "This is a terrible experience.",
        "The movie was okay, not great but not bad either."
    ]
    results = {text: analyzer.analyze(text) for text in sample_texts}
    for text, sentiment in results.items():
        print(f'Text: {text}\nSentiment: {sentiment}\n')

if __name__ == '__main__':
    main()