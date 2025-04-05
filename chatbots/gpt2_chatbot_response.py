import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Chatbot:
    def __init__(self):
        """Initializes the chatbot with a pre-trained GPT-2 model and tokenizer."""
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

    def generate_response(self, input_text):
        """Generates a response for the given input text using the GPT-2 model."""
        inputs = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=50, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

if __name__ == '__main__':
    chatbot = Chatbot()
    user_input = 'What is the capital of France?'
    response = chatbot.generate_response(user_input)
    print(f'User: {user_input}')
    print(f'Chatbot: {response}')