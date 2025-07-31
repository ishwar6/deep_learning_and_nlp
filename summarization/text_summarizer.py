import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

class Summarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text, max_length=130, min_length=30):
        inputs = self.tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

if __name__ == '__main__':
    text = 'Natural language processing (NLP) is a subfield of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to enable computers to understand, interpret, and produce human language in a valuable way.'
    summarizer = Summarizer()
    print(summarizer.summarize(text))