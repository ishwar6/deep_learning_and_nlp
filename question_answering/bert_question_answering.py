import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import pipeline

class QuestionAnsweringModel:
    def __init__(self):
        """Initializes the question answering model and tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    def answer_question(self, question, context):
        """Generates an answer to a question based on the provided context."""
        inputs = self.tokenizer(question, context, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        return answer

if __name__ == '__main__':
    context = "Transformers are a type of model architecture that is particularly effective for NLP tasks."
    question = "What are transformers?"
    qa_model = QuestionAnsweringModel()
    answer = qa_model.answer_question(question, context)
    print(f'Question: {question}\nAnswer: {answer}')