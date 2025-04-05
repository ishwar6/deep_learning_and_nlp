import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class SimpleGPT2Model:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def generate_text(self, prompt, max_length=50):
        tokens = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.generate(tokens, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == '__main__':
    gpt2_model = SimpleGPT2Model()
    prompt_text = 'Once upon a time'
    generated_text = gpt2_model.generate_text(prompt_text)
    print('Generated Text:', generated_text)