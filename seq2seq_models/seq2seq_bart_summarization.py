import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration

class SimpleDataset(Dataset):
    """Custom dataset for sequence-to-sequence tasks."""
    def __init__(self, texts, summaries):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.tokenizer(summary, return_tensors='pt', padding=True, truncation=True)
        return inputs.input_ids.squeeze(), outputs.input_ids.squeeze()

class Seq2SeqModel:
    """Sequence-to-sequence model encapsulating BART for summarization."""
    def __init__(self):
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)

    def train(self, dataloader, epochs=3):
        """Train the BART model on the provided dataset."""
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                inputs, targets = batch
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=inputs, labels=targets)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

    def generate_summary(self, text):
        """Generate a summary for the input text using the trained model."""
        self.model.eval()
        inputs = BartTokenizer.from_pretrained('facebook/bart-base').encode(text, return_tensors='pt')
        summary_ids = self.model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)
        return BartTokenizer.from_pretrained('facebook/bart-base').decode(summary_ids[0], skip_special_tokens=True)

if __name__ == '__main__':
    texts = ['The quick brown fox jumps over the lazy dog.'] * 10
    summaries = ['A fox jumps over a dog.'] * 10
    dataset = SimpleDataset(texts, summaries)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    seq2seq_model = Seq2SeqModel()
    seq2seq_model.train(dataloader)
    summary = seq2seq_model.generate_summary('The quick brown fox jumps over the lazy dog.')
    print('Generated Summary:', summary)