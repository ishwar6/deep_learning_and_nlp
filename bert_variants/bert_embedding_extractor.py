import torch
from transformers import BertTokenizer, BertModel

class BertInference:
    def __init__(self, model_name='bert-base-uncased'):
        """Initializes the BERT model and tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def encode_text(self, text):
        """Encodes text into BERT embeddings."""
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def get_embedding(self, text):
        """Fetches the first token's embedding for the given text."""
        embeddings = self.encode_text(text)
        return embeddings[0][0].numpy()

if __name__ == '__main__':
    bert_inference = BertInference()
    sample_text = "Hello, BERT!"
    embedding = bert_inference.get_embedding(sample_text)
    print(f'Embedding for \'{sample_text}\': {embedding}')