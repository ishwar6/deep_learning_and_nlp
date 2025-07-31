import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class SimpleGPT2:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_text(self, prompt, max_length=50):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def print_model_summary(self):
        return str(self.model)

if __name__ == '__main__':
    gpt2 = SimpleGPT2()
    prompt = 'In the future, AI will'
    generated_text = gpt2.generate_text(prompt)
    print('Generated Text:', generated_text)
    print('Model Summary:', gpt2.print_model_summary())