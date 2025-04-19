import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

class Summarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text, max_length=130, min_length=30):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
        summaries = self.model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(summaries[0], skip_special_tokens=True)

if __name__ == '__main__':
    sample_text = "The quick brown fox jumps over the lazy dog. This sentence is often used to demonstrate the use of all letters in the English alphabet."
    summarizer = Summarizer()
    summary = summarizer.summarize(sample_text)
    print('Summary:', summary)