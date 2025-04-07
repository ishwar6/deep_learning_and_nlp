import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BartTokenizer, BartForConditionalGeneration

class TextSummarizer:
    def __init__(self):
        """Initializes the text summarizer with a pre-trained BART model."""
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    def summarize(self, text):
        """Generates a summary of the input text."""
        inputs = self.tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

if __name__ == '__main__':
    sample_text = ("Deep learning models have made significant advancements in natural language processing tasks. "
                   "In particular, models like BART have shown effectiveness in summarization, translation, and more.")
    summarizer = TextSummarizer()
    summary = summarizer.summarize(sample_text)
    print('Original Text:', sample_text)
    print('Summary:', summary)