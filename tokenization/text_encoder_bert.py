from transformers import BertTokenizer, BertModel
import torch

class TextEncoder:
    """Encodes text using a pre-trained BERT model. """
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def encode(self, texts):
        """Tokenizes and encodes a list of texts into embeddings."""
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return model_output.last_hidden_state

if __name__ == '__main__':
    texts = ["Hello, how are you?", "I am learning about tokenization."]
    encoder = TextEncoder()
    embeddings = encoder.encode(texts)
    print(embeddings.shape)