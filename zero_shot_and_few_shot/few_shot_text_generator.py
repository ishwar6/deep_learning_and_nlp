import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class FewShotTextGenerator:
    def __init__(self, model_name='gpt2'):
        """Initializes the text generator with a pre-trained model and tokenizer."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    def generate_text(self, prompt, max_length=50):
        """Generates text given a prompt using the model."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

if __name__ == '__main__':
    text_generator = FewShotTextGenerator()
    prompt = 'Once upon a time in a land far away'
    generated_text = text_generator.generate_text(prompt)
    print(generated_text)