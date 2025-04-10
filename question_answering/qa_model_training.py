import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import Trainer, TrainingArguments

class QuestionAnsweringModel:
    def __init__(self):
        """Initializes the Question Answering model and tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    def tokenize(self, question, context):
        """Tokenizes the input question and context for the model."""
        inputs = self.tokenizer(question, context, return_tensors='pt')
        return inputs

    def train(self, train_dataset):
        """Trains the model on the provided dataset."""
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir='./logs',
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

    def predict(self, question, context):
        """Generates answer spans for a given question and context."""
        inputs = self.tokenize(question, context)
        outputs = self.model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)  
        answer_end = torch.argmax(outputs.end_logits)  
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end + 1]))

# Mock training dataset
class MockDataset:
    def __init__(self):
        self.data = [
            {'question': 'What is the capital of France?', 'context': 'The capital of France is Paris.'},
            {'question': 'What is the capital of Germany?', 'context': 'The capital of Germany is Berlin.'}
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    model = QuestionAnsweringModel()
    mock_dataset = MockDataset()
    model.train(mock_dataset)
    answer = model.predict('What is the capital of France?', 'The capital of France is Paris.')
    print(f'Predicted Answer: {answer}')