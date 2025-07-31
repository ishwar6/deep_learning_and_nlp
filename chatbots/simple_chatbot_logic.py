import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class SimpleChatbot:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def generate_response(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=50, num_return_sequences=1)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def chat(self):
        print("Chatbot: Hi! I'm a simple chatbot. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            response = self.generate_response(user_input)
            print(f"Chatbot: {response}")

if __name__ == '__main__':
    chatbot = SimpleChatbot()
    chatbot.chat()