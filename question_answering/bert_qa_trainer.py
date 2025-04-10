import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset

class QADataSet(Dataset):
    def __init__(self, questions, contexts, answers):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.questions[idx], self.contexts[idx], return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        start_positions = self.answers[idx]['start']
        end_positions = self.answers[idx]['end']
        inputs['start_positions'] = torch.tensor(start_positions)
        inputs['end_positions'] = torch.tensor(end_positions)
        return inputs

class QAModel:
    def __init__(self):
        self.model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

    def train(self, dataloader, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                print(f'Epoch: {epoch}, Loss: {loss.item()}')

    def predict(self, question, context):
        self.model.eval()
        inputs = self.model.tokenizer(question, context, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)
        return context[start_index:end_index + 1]

if __name__ == '__main__':
    questions = ['What is the capital of France?']
    contexts = ['The capital of France is Paris.']
    answers = [{'start': 30, 'end': 35}]
    dataset = QADataSet(questions, contexts, answers)
    dataloader = DataLoader(dataset, batch_size=1)
    qa_model = QAModel()
    qa_model.train(dataloader)
    answer = qa_model.predict(questions[0], contexts[0])
    print(f'Predicted answer: {answer}')