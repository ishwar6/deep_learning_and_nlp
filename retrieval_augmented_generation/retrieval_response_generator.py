import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class RetrievalAugmentedGenerator:
    def __init__(self, model_name='gpt2'):
        """Initializes the RetrievalAugmentedGenerator with a pretrained model and tokenizer."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_response(self, query, retrieval_context):
        """Generates a response based on the user's query and the retrieved context."""
        input_text = f'{retrieval_context} {query}'
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=150, num_return_sequences=1)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

if __name__ == '__main__':
    generator = RetrievalAugmentedGenerator()
    mock_query = 'What are the benefits of deep learning?'
    mock_context = 'Deep learning allows for automated feature extraction and improved accuracy.'
    result = generator.generate_response(mock_query, mock_context)
    print(result)