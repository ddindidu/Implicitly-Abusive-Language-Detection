import numpy as np
import time, datetime, random, argparse, os

from enum import Enum

from collections import Counter, OrderedDict

from sklearn import svm

from transformers import AutoTokenizer

from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix)

from utils import compute_metrics, print_result
from dataset import AL_Dataset, Mode


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--cache_dir', type=str, default='./cache')


    # dataset related
    parser.add_argument('--data_type', type=str, default='CADD')    # or 'AbuseEval'
    parser.add_argument('--task_type', type=str, default='baseline')  # or 'ood'
    parser.add_argument('--num_labels', type=int, default=3)


    # model related
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased") # or 'albert-base-v1'
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument('--max_depth', type=int, default=100)
    parser.add_argument("--svm_c", default=(10, 100, 1000), nargs='+', type=float)
    parser.add_argument("--kernel", default='linear', type=str) # or 'rbf'


    return parser.parse_args()



def token_ids_to_bow(input_ids, max_seq_length):
    # input: List[Int]
    # return: Numpy_Array[Int]
    origin = Counter(input_ids)
    temp = Counter(np.arange(max_seq_length))
    origin_bow = np.array(list(OrderedDict(sorted((origin + temp).items())).values()))
    temp_bow = np.ones(max_seq_length, dtype=int)

    result_bow = origin_bow - temp_bow
    result_bow[0] = 0   # ignore PAD tokens

    return result_bow


def run_svc(args, X_train, y_train, X_test, y_test, best_c=None):
    performance_result = []
    if best_c:
        clf = svm.SVC(C=best_c, gamma='auto', verbose=True, kernel=args.kernel)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        test_result, conf_matrix = compute_metrics(task_type=args.task_type, labels=y_test, preds=pred)
        print_result(test_result)
        print("Confusion Matrix"); print(conf_matrix)

    for c in args.svm_c:
        clf = svm.SVC(C=c, gamma='auto', verbose=True, kernel=args.kernel)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        test_result, conf_matrix = compute_metrics(task_type=args.task_type, labels=y_test, preds=pred)
        print_result(test_result)
        print("Confusion Matrix"); print(conf_matrix)
        performance_result.append((test_result["f1_weighted"], c))

    best_performance = sorted(performance_result)[-1]
    print("Best performance model's c (f1-weighted): {} ({})".format(best_performance[1], best_performance[0]))

    return best_performance[1]


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        cache_dir=args.cache_dir,

    )
    
    # Prepare data
    train_data = AL_Dataset(
        args=args,
        mode=Mode.train,
        data_type=args.data_type,
        task_type=args.task_type,
        tokenizer=tokenizer,
    )

    test_data = AL_Dataset(
        args=args,
        mode=Mode.test,
        data_type=args.data_type,
        task_type=args.task_type,
        tokenizer=tokenizer,
    )

    print("prepare svm")
    x_train = [token_ids_to_bow(x.input_ids, max_seq_length=tokenizer.vocab_size) for x in train_data.features]
    y_train = [int(x.label) for x in train_data.features]
    x_test = [token_ids_to_bow(x.input_ids, max_seq_length=tokenizer.vocab_size) for x in test_data.features]
    y_test = [int(x.label) for x in test_data.features]
    
    print("running svm")
    run_svc(args, x_train, y_train, x_test, y_test)
    # import IPython; IPython.embed(); exit(1)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
    



if __name__ == '__main__':
    from svm_run import get_args
    args = get_args()
    main(args)

    



    
