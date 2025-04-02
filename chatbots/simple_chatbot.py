import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class SimpleChatbot:
    def __init__(self):
        """Initializes the SimpleChatbot with a pre-trained model and tokenizer."""
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

    def generate_response(self, prompt, max_length=50):
        """Generates a response based on the input prompt using the GPT-2 model."""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

if __name__ == '__main__':
    chatbot = SimpleChatbot()
    user_input = 'Hello, how can I help you today?'
    response = chatbot.generate_response(user_input)
    print('Chatbot response:', response)