import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, model, tokenizer, max_length=50):
    """Generates text using a pre-trained model based on the input prompt."""
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated

def main():
    """Main function to set up model and generate text."""
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    prompt = 'Once upon a time,'
    generated_text = generate_text(prompt, model, tokenizer)
    print(generated_text)

if __name__ == '__main__':
    main()