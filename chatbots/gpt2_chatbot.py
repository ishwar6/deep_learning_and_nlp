import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Chatbot:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    def generate_response(self, prompt, max_length=50):
        tokens = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(tokens, max_length=max_length, num_return_sequences=1)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

if __name__ == '__main__':
    chatbot = Chatbot()
    prompt = 'What is the future of AI?'
    response = chatbot.generate_response(prompt)
    print('Chatbot response:', response)