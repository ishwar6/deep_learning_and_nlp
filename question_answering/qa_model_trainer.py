import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

class QuestionAnsweringModel:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    def encode_data(self, questions, contexts):
        inputs = self.tokenizer(questions, contexts, return_tensors='pt', padding=True, truncation=True)
        return inputs

    def train(self, train_dataset):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=10_000,
            save_total_limit=2,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

    def predict(self, question, context):
        inputs = self.encode_data([question], [context])
        with torch.no_grad():
            outputs = self.model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores) + 1
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_index:end_index]))

if __name__ == '__main__':
    model = QuestionAnsweringModel()
    dataset = load_dataset('squad', split='train[:1%]')
    model.train(dataset)
    question = 'What is the capital of France?'
    context = 'The capital of France is Paris.'
    answer = model.predict(question, context)
    print(f'Answer: {answer}')