import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

class Summarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text, max_length=130, num_beams=4):
        inputs = self.tokenizer.encode(text, return_tensors='pt')
        summary_ids = self.model.generate(inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

if __name__ == '__main__':
    text = "The quick brown fox jumps over the lazy dog. This sentence is often used as a placeholder text in various types of writing. It showcases the ability to use all the letters in the English alphabet in a single sentence."
    summarizer = Summarizer()
    summary = summarizer.summarize(text)
    print('Original Text:', text)
    print('Summarized Text:', summary)