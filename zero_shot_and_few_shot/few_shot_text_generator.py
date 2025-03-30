import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class FewShotTextGenerator:
    def __init__(self, model_name='gpt2'):
        """Initialize the model and tokenizer."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    def generate_text(self, prompt, max_length=50):
        """Generate text based on the provided prompt."""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    generator = FewShotTextGenerator()
    prompt = 'Once upon a time in a distant land'
    generated_text = generator.generate_text(prompt)
    print('Generated Text:', generated_text)