import spacy
from spacy.tokens import Doc
from transformers import DistilBertTokenizer, DistilBertModel
import torch

class TextEmbedding:
    def __init__(self, model_name='distilbert-base-uncased'):
        """Initializes the TextEmbedding class with a specified transformer model."""
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)

    def embed_text(self, text):
        """Embeds the input text using the transformer model and returns the embeddings."""
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def main():
    nlp = spacy.load('en_core_web_sm')
    text = "Deep learning is transforming the field of natural language processing."
    doc = nlp(text)
    text_embedding = TextEmbedding()
    embeddings = text_embedding.embed_text(doc.text)
    print(f'Embeddings for the text: {embeddings}')

if __name__ == '__main__':
    main()