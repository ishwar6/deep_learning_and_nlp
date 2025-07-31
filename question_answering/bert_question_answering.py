import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import pipeline

class QuestionAnsweringModel:
    def __init__(self, model_name='bert-base-uncased'):
        """Initializes the Question Answering model with the specified model name."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.qa_pipeline = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer)

    def answer_question(self, question, context):
        """Returns the answer to a question based on the provided context."""
        result = self.qa_pipeline(question=question, context=context)
        return result['answer']

if __name__ == '__main__':
    context = "The capital of France is Paris. Paris is known for its art, fashion, and culture."
    question = "What is the capital of France?"
    qa_model = QuestionAnsweringModel()
    answer = qa_model.answer_question(question, context)
    print(f'Question: {question}\nAnswer: {answer}')