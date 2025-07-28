from transformers import AutoTokenizer, AutoModel
import torch

class TextTokenizer:
    def __init__(self, model_name):
        """Initializes the tokenizer and model based on the provided model name."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def tokenize(self, text):
        """Tokenizes the input text and returns input IDs and attention mask."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return inputs['input_ids'], inputs['attention_mask']

    def get_embeddings(self, text):
        """Generates embeddings for the input text using the model."""
        input_ids, attention_mask = self.tokenize(text)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

if __name__ == '__main__':
    tokenizer = TextTokenizer('bert-base-uncased')
    sample_text = "Hello, how are you today?"
    embeddings = tokenizer.get_embeddings(sample_text)
    print(embeddings.shape)