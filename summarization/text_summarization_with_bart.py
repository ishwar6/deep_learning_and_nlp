import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

class Summarizer:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    def summarize(self, text, max_length=130, min_length=30):
        """Generates a summary for the given text using BART model."""
        inputs = self.tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

if __name__ == '__main__':
    text = "The quick brown fox jumps over the lazy dog. This classic sentence is often used as a pangram in English and is useful for demonstrating the capabilities of various fonts and keyboards."
    summarizer = Summarizer()
    summary = summarizer.summarize(text)
    print('Summary:', summary)