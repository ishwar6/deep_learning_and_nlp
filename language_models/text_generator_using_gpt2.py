import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class LanguageModel:
    def __init__(self, model_name='gpt2'):
        """Initializes the language model with a pre-trained GPT-2 model and tokenizer."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_text(self, prompt, max_length=50):
        """Generates text based on the input prompt up to max_length tokens."""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    model = LanguageModel()
    prompt = 'In the future, artificial intelligence will'
    generated_text = model.generate_text(prompt)
    print(generated_text)