import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EvaluationMetrics:
    def __init__(self, y_true, y_pred):
        """Initialize the EvaluationMetrics with true and predicted labels."""
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_accuracy(self):
        """Calculate accuracy of the predictions."""
        return accuracy_score(self.y_true, self.y_pred)

    def calculate_precision(self):
        """Calculate precision of the predictions."""
        return precision_score(self.y_true, self.y_pred, average='weighted')

    def calculate_recall(self):
        """Calculate recall of the predictions."""
        return recall_score(self.y_true, self.y_pred, average='weighted')

    def calculate_f1(self):
        """Calculate F1 score of the predictions."""
        return f1_score(self.y_true, self.y_pred, average='weighted')

if __name__ == '__main__':
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    metrics = EvaluationMetrics(y_true, y_pred)
    print(f'Accuracy: {metrics.calculate_accuracy()}')
    print(f'Precision: {metrics.calculate_precision()}')
    print(f'Recall: {metrics.calculate_recall()}')
    print(f'F1 Score: {metrics.calculate_f1()}')