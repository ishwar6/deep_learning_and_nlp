import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

class TextSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        """Initialize the TextSummarizer with a pre-trained BART model and tokenizer."""
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text, max_length=130, min_length=30):
        """Generate a summary for the provided text using the BART model."""
        inputs = self.tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

if __name__ == '__main__':
    input_text = "The quick brown fox jumps over the lazy dog. This is a demonstration of the summarization capabilities of the BART model which is pre-trained on large datasets."  
    summarizer = TextSummarizer()
    summary = summarizer.summarize(input_text)
    print('Summary:', summary)