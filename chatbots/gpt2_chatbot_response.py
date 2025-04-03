import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Chatbot:
    """
    A simple chatbot class leveraging GPT-2 for response generation.
    """
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

    def generate_response(self, prompt):
        """
        Generates a response based on the input prompt.
        """
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=50, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

if __name__ == '__main__':
    chatbot = Chatbot()
    user_input = 'Hello, how are you?'
    response = chatbot.generate_response(user_input)
    print('Chatbot response:', response)