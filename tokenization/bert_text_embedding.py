from transformers import BertTokenizer, BertModel
import torch

class TextTokenizer:
    def __init__(self, model_name='bert-base-uncased'):
        """Initializes the tokenizer and model from the specified pre-trained model."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def tokenize_text(self, text):
        """Tokenizes the input text and returns input IDs and attention mask."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return inputs['input_ids'], inputs['attention_mask']

    def get_embeddings(self, text):
        """Generates embeddings for the input text using the BERT model."""
        input_ids, attention_mask = self.tokenize_text(text)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

if __name__ == '__main__':
    text = "Deep learning is a subfield of machine learning."
    tokenizer = TextTokenizer()
    embeddings = tokenizer.get_embeddings(text)
    print(embeddings.shape)