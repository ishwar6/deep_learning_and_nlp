import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset

class QADataset(Dataset):
    def __init__(self, questions, contexts, answers):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.questions[idx], self.contexts[idx], return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        start_position = self.answers[idx]['start']
        end_position = self.answers[idx]['end']
        return {**encoding, 'start_positions': torch.tensor(start_position), 'end_positions': torch.tensor(end_position)}

class QAModel:
    def __init__(self):
        self.model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)

    def train(self, dataloader, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

questions = ['What is the capital of France?']
contexts = ['The capital of France is Paris.']
answers = [{'start': 30, 'end': 35}]

dataset = QADataset(questions, contexts, answers)
dataloader = DataLoader(dataset, batch_size=1)
model = QAModel()
model.train(dataloader)