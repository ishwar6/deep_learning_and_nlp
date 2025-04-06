import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

class Summarizer:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    def summarize(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

if __name__ == '__main__':
    text = "Deep learning is a subset of machine learning in artificial intelligence that is based on neural networks. It has gained immense popularity due to its ability to handle vast amounts of data and perform complex tasks with high accuracy. The rise of deep learning has revolutionized various fields such as computer vision, natural language processing, and robotics."
    summarizer = Summarizer()
    print(summarizer.summarize(text))