import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

class Summarizer:
    def __init__(self):
        """Initialize the Summarizer with a pre-trained BART model and tokenizer."""
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    def summarize(self, text, max_length=130, min_length=30):
        """Generate a summary for the given text."""
        inputs = self.tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

if __name__ == '__main__':
    text = "Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals."  
    summarizer = Summarizer()
    summary = summarizer.summarize(text)
    print('Summary:', summary)