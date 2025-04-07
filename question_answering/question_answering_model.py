import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import Trainer, TrainingArguments

class QAData:
    def __init__(self, questions, contexts, answers):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers

    def tokenize(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(self.questions, self.contexts, return_tensors='pt', padding=True, truncation=True)
        return inputs

class QAModel:
    def __init__(self):
        self.model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    def train(self, inputs, answers):
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.start_logits.argmax(), outputs.end_logits.argmax()

if __name__ == '__main__':
    questions = ['What is the capital of France?']
    contexts = ['The capital of France is Paris.']
    answers = ['Paris']
    qa_data = QAData(questions, contexts, answers)
    tokenized_inputs = qa_data.tokenize()
    qa_model = QAModel()
    qa_model.train(tokenized_inputs, answers)
    start, end = qa_model.predict(tokenized_inputs)
    print(f'Predicted answer spans from {start} to {end}')