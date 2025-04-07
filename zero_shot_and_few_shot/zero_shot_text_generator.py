import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class ZeroShotTextGenerator:
    def __init__(self, model_name='gpt2'):
        """Initializes the text generator with a pre-trained model."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    def generate_text(self, prompt, max_length=50):
        """Generates text based on a given prompt using the model."""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    generator = ZeroShotTextGenerator()
    prompt = 'The future of AI in education is'
    generated_text = generator.generate_text(prompt)
    print(generated_text)