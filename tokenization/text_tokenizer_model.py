from transformers import BertTokenizer, BertModel
import torch

class TextTokenizer:
    def __init__(self, model_name='bert-base-uncased'):
        """Initialize the tokenizer and model."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def tokenize_text(self, text):
        """Tokenize input text and return input IDs and attention mask."""
        return self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    def get_embeddings(self, text):
        """Get embeddings for the tokenized text using the BERT model."""
        inputs = self.tokenize_text(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

if __name__ == '__main__':
    sample_text = "Hello, how are you today?"
    tokenizer = TextTokenizer()
    embeddings = tokenizer.get_embeddings(sample_text)
    print(embeddings.shape)