import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class RetrievalAugmentedGenerator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate(self, query, retrieval_context):
        combined_input = f'{retrieval_context} {query}'
        inputs = self.tokenizer(combined_input, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=150)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    rag = RetrievalAugmentedGenerator()
    mock_query = 'What are the benefits of deep learning?'
    mock_context = 'Deep learning allows for complex data representation and improved accuracy.'
    result = rag.generate(mock_query, mock_context)
    print(result)