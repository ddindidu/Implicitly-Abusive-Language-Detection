from transformers import BertModel, BertConfig
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoConfig, BertConfig


class ALDBert(nn.Module):
    def __init__(self, args, config):
        super(ALDBert, self).__init__()
        self.args = args
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.args.num_labels)


    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        _, pooled_output = self.bert(input_ids= input_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids).last_hidden_state
        logits = self.classifier(pooled_output)

        loss = None
        if self.args.num_labels > 3:    # baseline
            loss_func = nn.BCEWithLogitsLoss()
            loss = loss_func(logits, labels)
        elif self.args.num_labels == 2: # ood
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits.view(-1, self.args.num_labels), labels.view(-1))


        return SequenceClassifierOutput(
            loss = loss,
            logits = logits
            # When the model returns logits,
            # F.softmax(logits) converts logits to probability, outside
        )