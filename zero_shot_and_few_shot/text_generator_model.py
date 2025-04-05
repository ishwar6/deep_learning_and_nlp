import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def generate_text(prompt, model, tokenizer, max_length=50):
    """
    Generate text using a pre-trained language model.
    
    Args:
        prompt (str): The input text prompt for text generation.
        model: The pre-trained language model.
        tokenizer: The tokenizer for the model.
        max_length (int): The maximum length of generated text.
    
    Returns:
        str: Generated text.
    """
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    """
    Main function to execute text generation.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    prompt = "Once upon a time"
    generated_text = generate_text(prompt, model, tokenizer)
    print("Generated Text:", generated_text)


if __name__ == '__main__':
    main()