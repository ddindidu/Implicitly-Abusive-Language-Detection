import numpy as np
import argparse, time
import os
from typing import Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          AutoConfig,
                          set_seed,
                          )

from sklearn.metrics import (classification_report,
                             f1_score,
                             precision_score,
                             recall_score,
                             accuracy_score,
                             confusion_matrix,
                             )

from utils import format_time, compute_metrics, print_result
from dataset import AL_Dataset, Mode


def get_args():
    parser = argparse.ArgumentParser()

    # initialization
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    # dataset
    parser.add_argument('--data_type', type=str, default='CADD')    # or 'AbuseEval'
    parser.add_argument('--num_labels', type=int, default=3)    # or '2'

    # model
    parser.add_argument('--mode', type=str, default='train')    # or 'test'
    parser.add_argument('--task_type', type=str, default='baseline')    # or 'ood'
    parser.add_argument('--threshold', type=float, default = 0.8)
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-cased')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)

    # optimizer & scheduler
    parser.add_argument('--betas', type=float, default=(0.9, 0.98), nargs='+')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=int, default=2.4e4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--eps', type=float, default=1e-6)

    # trainer
    parser.add_argument('--load_from_checkpoint', type=str, default="")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--log_scale', type=int, default=10)
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('fast_dev_run', action='store_true', default=True)
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_test', action='store_true', default=True)

    return parser.parse_args()


def main(args):
    print(args)
    set_seed(args.seed) # 42
    writer = SummaryWriter(args.log_dir)

    device = torch.device('cuda')

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir = args.cache_dir,
    )

    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labals = args.num_labels,
        finetuning_task = args.task_type,
        cache_dir = args.cache_dir,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        num_labels=args.num_labels,
    )
    model.cuda()


    # Prepare data
    train_data = AL_Dataset(
        args = args,
        mode = Mode.train,
        data_type= args.data_type,
        task_type = args.task_type,
        tokenizer = tokenizer,
    )
    valid_data = AL_Dataset(
        args = args,
        mode = Mode.valid,
        data_type=args.data_type,
        task_type = args.task_type,
        tokenizer = tokenizer,
    )
    test_data = AL_Dataset(
        args = args,
        mode = Mode.test,
        data_type=args.data_type,
        task_type = args.task_type,
        tokenizer = tokenizer,
    )


    # Load data
    train_dl = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    valid_dl = DataLoader(
        dataset=valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_dl = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )


    # optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, '{}_{}_{}_{}.pt'.format(args.data_type,
                                                                          args.task_type,
                                                                          args.model_name_or_path,
                                                                          args.epochs))

    # call checkpoint if it exists.
    # please do not call checkpoint like this and change below this code when you are on developing process.
    if args.mode == 'test':
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            #optimizer.load_state_dict(ckpt['optimizer'])
            #last_epoch=ckpt['epoch']+1
            #print('Last checkpoint is loaded. start_epoch:', last_epoch)
            print('Loaded checkpoint: %s'%ckpt_path)
        else:
            raise FileNotFoundError('No checkpoint is found.')


    # Initialize gradients
    model.zero_grad()

    total_train_step, total_valid_step = 0, 0

    # Training & Validating
    if args.mode == 'train':
        for epoch_i in range(0, args.epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
            print('Training...')

            t0 = time.time()

            total_loss = 0

            # Training
            model.train()

            all_preds, all_labels = [], []
            loss_for_logging = 0
            for step, batch in enumerate(train_dl):

                total_train_step += 1
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, token_type_ids, labels = batch
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }

                # forward
                outputs = model(**inputs, labels = labels)
                loss, logits = outputs[0], outputs[1]
                total_loss += loss.item()
                loss_for_logging += loss.item()
                preds = logits.argmax(-1)

                # logging
                if step % args.log_scale == 0 and not step == 0:
                    writer.add_scalar('Train/loss', (loss_for_logging / args.log_scale), total_train_step)
                    loss_for_logging = 0

                if len(all_preds) == 0:
                    all_preds = preds.detach().cpu().clone().numpy()
                    all_labels = labels.detach().cpu().clone().numpy()
                else:
                    all_preds = np.append(all_preds, preds.detach().cpu().clone().numpy())
                    all_labels = np.append(all_labels, labels.detach().cpu().clone().numpy())

                if step % 500 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {}'.format(step, len(train_dl), elapsed, loss))


                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            print('total train loss: {}'.format(total_loss / len(train_dl)))
            train_result, confusion_matrix = compute_metrics(task_type = args.task_type, labels = all_labels, preds = all_preds)
            print_result(train_result)
            print("Confusion Matrix"); print(confusion_matrix)
            print('    Train epoch took: {:}'.format(format_time(time.time() - t0)))



            # Validating
            print("")
            print("Running Validation...")
            all_preds = []
            all_labels = []
            loss_for_logging = 0
            for step, batch in enumerate(valid_dl):
                total_valid_step += 1
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, token_type_ids, labels = batch

                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }

                with torch.no_grad():
                    outputs = model(**inputs)


                # loss = outputs.loss
                logits = outputs.logits
                preds = logits.argmax(-1) # [batch]
                # loss_for_logging += loss

                # if the maximum softmax probability is lower than threshold,
                # the data is considered as ood data
                # '''
                if args.task_type == 'ood':
                    probabilities = F.softmax(logits, dim=1)   # [batch, num_label]
                    max_prob = probabilities.max(-1).values # [batch]
                    # check up
                    # import IPython; IPython.embed(); exit(1)
                    for idx in range(len(max_prob)):    # iterate for one batch
                        if max_prob[idx] < args.threshold:
                            preds[idx] = 2
                # '''

                # logging
                #if step % args.log_scale == 0 and not step == 0:
                #    writer.add_scalar('Valid/loss', (loss_for_logging / args.log_scale), total_valid_step)
                #    loss_for_logging = 0

                if len(all_preds) == 0:
                    all_preds = preds.detach().cpu().clone().numpy()
                    all_labels = labels.detach().cpu().clone().numpy()
                else:
                    all_preds = np.append(all_preds, preds.detach().cpu().clone().numpy())
                    all_labels = np.append(all_labels, labels.detach().cpu().clone().numpy())

            print("")
            val_result, confusion_matrix = compute_metrics(task_type=args.task_type, labels=all_labels, preds=all_preds)
            print_result(val_result)
            print("Confusion Matrix");  print(confusion_matrix)
            print("  Validation epoch took: {:}".format(format_time(time.time() - t0)))

        print("")
        print("Training complete\n\n")

        # save model state
        ckpt = {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch_i}
        torch.save(ckpt, ckpt_path)
        print('Save the checkpoint: %s'%ckpt_path)




    # Test
    t0 = time.time()

    model.eval()
    all_preds = []
    all_labels = []
    all_probabilities = []

    for step, batch in enumerate(test_dl):
        if step % 100 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dl), elapsed))

        batch = tuple(t.to(device) for t in batch)

        input_ids, attention_mask, token_type_ids, labels = batch

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        preds = logits.argmax(-1)  # [batch]

        # for printing probabilities
        probabilities = F.softmax(logits, dim=1)    # [batch, num_label]

        # if the maximum softmax probability is lower than threshold,
        # the data is considered as ood data
        # '''
        if args.task_type == 'ood':
            probabilities = F.softmax(logits, dim=1)  # [batch, num_label]
            max_prob = probabilities.max(-1).values  # [batch]
            for idx in range(len(max_prob)):  # iterate for one batch
                if max_prob[idx] < args.threshold:
                    preds[idx] = 2
        # '''

        if len(all_preds) == 0:
            all_preds = preds.detach().cpu().clone().numpy()
            all_labels = labels.detach().cpu().clone().numpy()
            all_probabilities = probabilities.detach().cpu().clone().numpy()
        else:
            all_preds = np.append(all_preds, preds.detach().cpu().clone().numpy())
            all_labels = np.append(all_labels, labels.detach().cpu().clone().numpy())
            all_probabilities = np.vstack((all_probabilities, probabilities.detach().cpu().clone().numpy()))

    test_result, confusion_matrix = compute_metrics(task_type=args.task_type, labels=all_labels, preds=all_preds)
    print_result(test_result)
    print("Confusion Matrix");  print(confusion_matrix)
    print("  Test took: {:}".format(format_time(time.time() - t0)))
    print("{} ends.".format(args.task_type))

    #import IPython; IPython.embed(); exit(1)

    '''
    # labels and predicted probabilities
    # General Model (BERT)
    df_labels = pd.DataFrame(all_labels, columns = ['labels'])
    df_probabilities = pd.DataFrame(all_probabilities, columns = ['Not', 'Exp', 'Imp'])
    df_concat = pd.concat([df_labels, df_probabilities], axis=1)
    df_concat.to_csv('./analysis/results/{}_{}_{}_labels_and_probabilities.csv'.format(args.data_type, args.task_type, args.model_name_or_path), index=False)
    '''

    '''
    # labels and predicted probabilities
    # OOD Model (BERT+OOD)
    df_labels = pd.DataFrame(all_labels, columns=['labels'])
    df_probabilities = pd.DataFrame(all_probabilities, columns=['Not', 'Exp'])
    df_concat = pd.concat([df_labels, df_probabilities], axis=1)
    df_concat.to_csv(
        './analysis/results/{}_{}_{}_labels_and_probabilities.csv'.format(args.data_type, args.task_type, args.model_name_or_path),
        index=False)
    print("Save labels and probabilities file")
    '''