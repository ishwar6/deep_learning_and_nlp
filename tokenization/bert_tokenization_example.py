from transformers import BertTokenizer, BertModel
import torch

class TextTokenizer:
    def __init__(self, model_name='bert-base-uncased'):
        """Initializes the tokenizer and model."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def tokenize(self, text):
        """Tokenizes input text and returns input IDs and attention mask."""
        encoding = self.tokenizer(text, return_tensors='pt')
        return encoding['input_ids'], encoding['attention_mask']

    def get_embeddings(self, input_ids, attention_mask):
        """Obtains embeddings for the given input IDs and attention mask."""
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state

if __name__ == '__main__':
    sample_text = 'Deep learning models are transforming education.'
    tokenizer = TextTokenizer()
    input_ids, attention_mask = tokenizer.tokenize(sample_text)
    embeddings = tokenizer.get_embeddings(input_ids, attention_mask)
    print('Input IDs:', input_ids)
    print('Attention Mask:', attention_mask)
    print('Embeddings Shape:', embeddings.shape)