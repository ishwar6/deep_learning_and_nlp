import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import Trainer, TrainingArguments

class QuestionAnsweringModel:
    def __init__(self):
        """Initializes the BERT model for question answering."""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    def preprocess(self, question, context):
        """Tokenizes the input question and context."""
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
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

    def predict(self, question, context):
        """Generates an answer from the model based on the question and context."""
        inputs = self.preprocess(question, context)
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer_start = outputs.start_logits.argmax()  
        answer_end = outputs.end_logits.argmax() + 1  
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        return answer

if __name__ == '__main__':
    model = QuestionAnsweringModel()
    context = "Hugging Face is creating a tool that democratizes AI."
    question = "What is Hugging Face creating?"
    answer = model.predict(question, context)
    print(f'Answer: {answer}')