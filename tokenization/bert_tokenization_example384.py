from transformers import BertTokenizer, BertModel
import torch

class TokenizationExample:
    def __init__(self, model_name='bert-base-uncased'):
        """Initializes the tokenizer and model based on the specified BERT model name."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def tokenize_input(self, text):
        """Tokenizes the input text and returns input IDs and attention masks."""
        encoded = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return encoded['input_ids'], encoded['attention_mask']

    def get_embeddings(self, text):
        """Generates embeddings for the tokenized input text."""
        input_ids, attention_mask = self.tokenize_input(text)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

if __name__ == '__main__':
    sample_text = "Tokenization is a crucial step in NLP."
    tokenizer_example = TokenizationExample()
    embeddings = tokenizer_example.get_embeddings(sample_text)
    print(embeddings.shape)