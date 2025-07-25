import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EvaluationMetrics:
    def __init__(self, y_true, y_pred):
        """
        Initializes the EvaluationMetrics with true and predicted labels.
        :param y_true: List of true labels
        :param y_pred: List of predicted labels
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def accuracy(self):
        """
        Calculates the accuracy of predictions.
        :return: Accuracy score
        """
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self):
        """
        Calculates the precision of predictions.
        :return: Precision score
        """
        return precision_score(self.y_true, self.y_pred, average='weighted')

    def recall(self):
        """
        Calculates the recall of predictions.
        :return: Recall score
        """
        return recall_score(self.y_true, self.y_pred, average='weighted')

    def f1(self):
        """
        Calculates the F1 score of predictions.
        :return: F1 score
        """
        return f1_score(self.y_true, self.y_pred, average='weighted')

if __name__ == '__main__':
    true_labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    predicted_labels = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    evaluator = EvaluationMetrics(true_labels, predicted_labels)
    print('Accuracy:', evaluator.accuracy())
    print('Precision:', evaluator.precision())
    print('Recall:', evaluator.recall())
    print('F1 Score:', evaluator.f1())