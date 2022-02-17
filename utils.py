import csv
import time, datetime, random, os

from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, confusion_matrix)

def f1_pre_rec_scalar(labels, preds, main_label=1):
    return {
        "acc": simple_accuracy(labels, preds),
        "precision_micro": precision_score(labels, preds, average="micro"),
        "recall_micro": recall_score(labels, preds, average="micro"),
        "f1_micro": f1_score(labels, preds, average="micro"),
        "precision_weighted": precision_score(labels, preds, average="weighted"),
        "recall_weighted": recall_score(labels, preds, average="weighted"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average=None)[main_label],
        "recall": recall_score(labels, preds, average=None)[main_label],
        "f1": f1_score(labels, preds, average=None)[main_label],
        "f1_macro": f1_score(labels, preds, average="macro"),
    }, confusion_matrix(labels, preds)


def compute_metrics(task_type, labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec_scalar(labels, preds)


def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def print_result(test_result):
    for name, value in test_result.items():
        print('   Average  ' +name, value)
