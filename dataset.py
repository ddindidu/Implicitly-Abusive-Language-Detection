import os
import torch
import csv, json, copy
from enum import Enum
from typing import Union

from torch.utils.data import Dataset, DataLoader


class Mode(Enum):
    train = "train"
    valid = "valid"
    test = "test"


class InputExample(object):
	"""
	A single training/test example for simple sequence classification.
	"""

	def __init__(self, guid, text_a, text_b=None, label=None, features=[]):
	    self.guid = guid
	    self.label = label
	    self.text_a = text_a
	    self.text_b = text_b
	    self.features = features

	def __repr__(self):
		return str(self.to_json_string())

	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class AL_Dataset(Dataset):
    def __init__(self, args, mode=Union[str, Mode], task_type='baseline', data_type='CADD', tokenizer=None):
        self.args = args
        self.task_type = task_type
        self.data_type = data_type

        if isinstance(mode, str):
            try:
                mode=Mode[mode]
            except KeyError as e:
                raise KeyError("mode is not a valid name")

        cached_features_file = os.path.join(
            args.cache_dir if args.cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}_{}".format(mode.value,
                                        tokenizer.__class__.__name__,
                                        str(args.max_seq_len),
                                        args.data_type,
                                        args.task_type,
                                        ),
        )

        #if args.test_dir == True:

        if data_type == 'CADD':
            path = args.data_dir+'/CADD'
        elif data_type == 'AbuseEval':
            path = args.data_dir+'/AbuseEval'

        if task_type == 'baseline':
            path = path+'/baseline/'
            self.label_list = ['0', '1', '2']
        elif task_type == 'ood':
            path = path+'/ood/'
            self.label_list = ['0', '1', '2']

        file_path = os.path.join(path, mode.value)+'.csv'
        if os.path.exists(file_path):
            #print("File path: %s"%file_path)
            pass
        else:
            raise FileNotFoundError

        file = read_file(data_type, file_path)

        if data_type == 'CADD':
            contexts, comments, labels = file[0], file[1], file[2]
            #contexts, comments, labels = file[0][:100], file[1][:100], file[2][:100]
        elif data_type == 'AbuseEval':
            contexts, comments, labels = [], file[0], file[1]

        examples = generate_examples(mode, data_type, contexts, comments, labels)
        self.features = convert_examples_to_features(
            args=args,
            label_list=self.label_list,
            examples=examples,
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
        )
        torch.save(self.features, cached_features_file)


    def __len__(self):
        return len(self.features)


    def __getitem__(self, idx):
        attention_mask = self.features[idx].attention_mask
        input_ids = self.features[idx].input_ids
        token_type_ids = self.features[idx].token_type_ids
        label = self.features[idx].label
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids), torch.tensor(label)

    def get_labels(self):
        return self.labels


def read_file(data_type='CADD', path=""):
    rows = []
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',', lineterminator='\n')
        next(reader)    # skip the first header row of the CSV file
        for row in reader:
            rows.append(row)

    if data_type == 'CADD':
        contexts, comments, labels, lenComments, lenContexts = [r for r in zip(*rows)]
        assert len(contexts) == len(comments) == len(labels) == len(lenComments) == len(lenContexts)
        return [contexts, comments, labels]
    elif data_type == 'AbuseEval':
        ids, comments, labels = [r for r in zip(*rows)]
        assert len(comments) == len(labels)
        return [comments, labels]


def generate_examples(mode, data_type='CADD', contexts=[], comments=[], labels=[]):
    examples = []

    if data_type == 'CADD':
        for idx in range(len(contexts)):
            guid = "%s-%s"%(mode, idx)
            text_a = contexts[idx]
            text_b = comments[idx]
            label = labels[idx]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
    elif data_type == 'AbuseEval':
        for idx in range(len(comments)):
            guid = "%s-%s"%(mode, idx)
            text_a = comments[idx]
            label = labels[idx]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )

    return examples





def convert_examples_to_features(args, label_list, examples, tokenizer, max_length):
    output_mode = 'classification'


    # prepare labels
    label_map = {label: i for i, label in enumerate(label_list)}
    def label_from_example(label):
        if output_mode == "classification":
            return label_map[label]
        elif output_mode == "regression":
            return float(label)
        raise KeyError(output_mode)
    labels = [label_from_example(example.label) for example in examples]


    # prepare tokenizer
    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) if example.text_b else example.text_a for example in examples],
        max_length = max_length,
        padding = 'max_length',
        truncation = 'longest_first',
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    return features