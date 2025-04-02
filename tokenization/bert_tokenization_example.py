from transformers import BertTokenizer, BertModel
import torch

class TokenizationExample:
    def __init__(self, model_name='bert-base-uncased'):
        """Initializes the tokenizer and model."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def tokenize_input(self, text):
        """Tokenizes the input text and returns input IDs and attention masks."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return inputs['input_ids'], inputs['attention_mask']

    def get_embeddings(self, text):
        """Generates embeddings for the input text using BERT model."""
        input_ids, attention_mask = self.tokenize_input(text)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

if __name__ == '__main__':
    example = TokenizationExample()
    text = "Hello, how are you today?"
    embeddings = example.get_embeddings(text)
    print(embeddings.shape)