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

    def preprocess_data(self, dataset):
        def encode(examples):
            return self.tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)
        return dataset.map(encode, batched=True)

    def train(self, train_dataset):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_dir='./logs',
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

    def predict(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores) + 1
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_index:end_index]))

if __name__ == '__main__':
    model = QuestionAnsweringModel()
    dataset = load_dataset('squad')
    train_data = model.preprocess_data(dataset['train'])
    model.train(train_data)
    question = 'What is the capital of France?'
    context = 'The capital of France is Paris.'
    answer = model.predict(question, context)
    print(f'Answer: {answer}')