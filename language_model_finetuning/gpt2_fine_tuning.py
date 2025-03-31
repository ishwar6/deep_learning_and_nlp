import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class FineTuner:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.train()

    def tokenize_data(self, texts):
        """Tokenizes input texts into model-readable format."""
        return self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    def train(self, texts, labels, epochs=3, lr=5e-5):
        """Trains the model on given texts and labels."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            inputs = self.tokenize_data(texts)
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def generate_text(self, prompt, max_length=50):
        """Generates text based on the input prompt."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == '__main__':
    fine_tuner = FineTuner()
    sample_texts = ['Hello, how are you?', 'What is the weather like today?']
    sample_labels = sample_texts  # In a real scenario, these would be different
    fine_tuner.train(sample_texts, sample_labels)
    generated = fine_tuner.generate_text('What will happen next?')
    print(f'Generated Text: {generated}')