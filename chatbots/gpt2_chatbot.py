import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Chatbot:
    """
    A simple chatbot class using the GPT-2 model from Hugging Face Transformers.
    """

    def __init__(self):
        """
        Initializes the chatbot with a pre-trained GPT-2 model and tokenizer.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def generate_response(self, user_input):
        """
        Generates a response to the user input using the GPT-2 model.
        
        Args:
            user_input (str): The input text from the user.
        
        Returns:
            str: The generated response from the chatbot.
        """
        input_ids = self.tokenizer.encode(user_input, return_tensors='pt')
        output = self.model.generate(input_ids, max_length=50, num_return_sequences=1)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

if __name__ == '__main__':
    chatbot = Chatbot()
    user_input = 'Hello, how are you today?'
    response = chatbot.generate_response(user_input)
    print('Chatbot response:', response)