import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

class TextSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        """Initialize the TextSummarizer with a pre-trained BART model."""
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text, max_length=130, min_length=30, length_penalty=2.0, num_beams=4):
        """Generate a summary for the given text using BART model."""
        inputs = self.tokenizer.encode(text, return_tensors='pt')
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=min_length,
                                           length_penalty=length_penalty, num_beams=num_beams,
                                           early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == '__main__':
    text = "Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals."
    summarizer = TextSummarizer()
    summary = summarizer.summarize(text)
    print('Summary:', summary)