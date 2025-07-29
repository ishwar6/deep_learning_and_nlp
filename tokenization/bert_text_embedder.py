from transformers import BertTokenizer, BertModel
import torch

class TextEmbedder:
    """
    A class to tokenize text and generate embeddings using BERT.
    """

    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def tokenize(self, text):
        """
        Tokenizes input text into BERT format.
        """
        return self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    def get_embeddings(self, text):
        """
        Generates embeddings for the input text.
        """
        inputs = self.tokenize(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

if __name__ == '__main__':
    embedder = TextEmbedder()
    sample_text = "Hello, how are you today?"
    embeddings = embedder.get_embeddings(sample_text)
    print(embeddings)