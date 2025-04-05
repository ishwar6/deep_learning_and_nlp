import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


def load_and_prepare_data():
    """Loads and prepares the IMDB dataset for sentiment analysis."""
    dataset = load_dataset('imdb')
    return dataset


def tokenize_data(dataset, tokenizer):
    """Tokenizes the dataset using the provided tokenizer."""
    return dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True), batched=True)


def train_model(train_dataset):
    """Trains the BERT model on the provided training dataset."""
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir='./logs',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()
    return model


def main():
    """Main function to execute data loading, tokenization, and model training."""
    dataset = load_and_prepare_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_dataset = tokenize_data(dataset['train'], tokenizer)
    model = train_model(tokenized_dataset)
    print("Model training complete!")


if __name__ == '__main__':
    main()