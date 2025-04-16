import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

class QuestionAnsweringModel:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    def preprocess_data(self, dataset):
        def preprocess_function(examples):
            inputs = self.tokenizer(examples['question'], examples['context'], max_length=512, truncation=True)
            return inputs
        return dataset.map(preprocess_function, batched=True)

    def train(self, train_dataset):
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

    def predict(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors='pt')
        outputs = self.model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores) + 1
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_index:end_index]))
        return answer

if __name__ == '__main__':
    dataset = load_dataset('squad')
    qa_model = QuestionAnsweringModel()
    train_dataset = qa_model.preprocess_data(dataset['train'])
    qa_model.train(train_dataset)
    question = 'What is the capital of France?'
    context = 'The capital of France is Paris.'
    answer = qa_model.predict(question, context)
    print(f'Answer: {answer}')