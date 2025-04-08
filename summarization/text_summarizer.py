import torch
from transformers import BartTokenizer, BartForConditionalGeneration

class Summarizer:
    def __init__(self):
        """Initializes the model and tokenizer for summarization."""
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    def summarize(self, text):
        """Generates a summary for the given text."""
        inputs = self.tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

if __name__ == '__main__':
    text = "Deep learning is a subset of machine learning that deals with neural networks. It has gained immense popularity due to its ability to analyze large amounts of data and produce powerful models for tasks like image recognition and natural language processing."
    summarizer = Summarizer()
    summary = summarizer.summarize(text)
    print('Summary:', summary)