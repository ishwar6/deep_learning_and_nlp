import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TextGenerationModel:
    def __init__(self, model_name='gpt2'):
        """Initializes the text generation model with a specified transformer model."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_text(self, prompt, max_length=50):
        """Generates text based on the input prompt."""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    model = TextGenerationModel()
    prompt = 'In a future world where technology has advanced,'
    generated_text = model.generate_text(prompt)
    print(generated_text)