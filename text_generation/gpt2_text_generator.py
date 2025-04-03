import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TextGenerator:
    def __init__(self, model_name='gpt2'):
        """Initializes the TextGenerator with a pre-trained model and tokenizer."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    def generate_text(self, prompt, max_length=50):
        """Generates text given a prompt until the maximum length is reached."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == '__main__':
    text_generator = TextGenerator()
    prompt = 'Once upon a time'
    generated_text = text_generator.generate_text(prompt)
    print(generated_text)