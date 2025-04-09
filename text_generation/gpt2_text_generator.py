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

    def fine_tune(self, train_data, epochs=1):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
        for epoch in range(epochs):
            for text in train_data:
                inputs = self.tokenizer.encode(text, return_tensors='pt')
                labels = inputs.clone()
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

if __name__ == '__main__':
    generator = TextGenerator()
    print(generator.generate_text('Once upon a time', max_length=30))
    mock_train_data = ['The quick brown fox jumps over the lazy dog.', 'Deep learning is fascinating.']
    generator.fine_tune(mock_train_data, epochs=2)