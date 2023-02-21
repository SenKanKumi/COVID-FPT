import numpy as np
import torch
from sklearn.metrics import accuracy_score


def Confusion_matrix(pred, target, matrix):
    for (t, p) in zip(target, pred):
        matrix[t, p] += 1
    return matrix


def benchmark(matrix):
    TP = np.diag(matrix)
    FP = np.sum(np.array(matrix), axis=0) - TP
    FN = np.sum(np.array(matrix), axis=1) - TP
    TN = np.sum(np.sum(np.array(matrix), axis=0)) - (TP + FP + FN)
    TP = TP.astype(float)
    TN = TN.astype(float)
    FP = FP.astype(float)
    FN = FN.astype(float)
    Accuracy = TP / (TP + FP + FN + TN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    return Accuracy, Precision, Recall


if __name__ == "__main__":
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    p = torch.tensor([0, 2, 1, 0, 0, 1])
    matrix = torch.zeros(3, 3)
    matrix = Confusion_matrix(p, y, matrix)
    benchmark(matrix)
