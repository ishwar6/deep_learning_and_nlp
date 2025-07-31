import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextGenerator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_text(self, prompt, max_length=50):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def train(self, dataset, epochs=1, learning_rate=5e-5):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        for epoch in range(epochs):
            for text in dataset:
                inputs = self.tokenizer.encode(text, return_tensors='pt')
                labels = inputs.clone()
                optimizer.zero_grad()
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

if __name__ == '__main__':
    sample_dataset = [
        'Once upon a time in a land far away.',
        'The quick brown fox jumps over the lazy dog.',
        'To be or not to be, that is the question.'
    ]
    generator = TextGenerator()
    generator.train(sample_dataset, epochs=2)
    generated_text = generator.generate_text('In a world where technology', max_length=30)
    print(generated_text)