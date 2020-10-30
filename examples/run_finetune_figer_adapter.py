# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" k-adapter for FIGER"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os,time
import random

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import json
import sys
import shutil

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  RobertaModel,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)
from pytorch_transformers.modeling_roberta import gelu

from pytorch_transformers.my_modeling_roberta import RobertaForFIGER
from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_figer import (compute_metrics, convert_examples_to_features_entity_typing,
                        output_modes, processors)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForFIGER, RobertaTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    pretrained_model = model[0]
    figer_model = model[1]
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    if args.freeze_bert:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in figer_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in figer_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in figer_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in figer_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in pretrained_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if args.freeze_bert:
            figer_model, optimizer = amp.initialize(figer_model, optimizer, opt_level=args.fp16_opt_level)
        else:
            figer_model, optimizer = amp.initialize(figer_model, optimizer, opt_level=args.fp16_opt_level)
            pretrained_model, optimizer = amp.initialize(pretrained_model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        if args.freeze_bert:
            figer_model = torch.nn.DataParallel(figer_model)
        else:
            pretrained_model = torch.nn.DataParallel(pretrained_model)
            figer_model = torch.nn.DataParallel(figer_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        if args.freeze_bert:
            figer_model = torch.nn.parallel.DistributedDataParallel(figer_model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
        else:
            figer_model = torch.nn.parallel.DistributedDataParallel(figer_model, device_ids=[args.local_rank],
                                                                     output_device=args.local_rank,
                                                                     find_unused_parameters=True)
            pretrained_model = torch.nn.parallel.DistributedDataParallel(pretrained_model, device_ids=[args.local_rank],
                                                                     output_device=args.local_rank,
                                                                     find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    logger.info("Try resume from checkpoint")
    if args.restore:
        if os.path.exists(os.path.join(args.output_dir, 'global_step.bin')):
            logger.info("Load last checkpoint data")
            global_step = torch.load(os.path.join(args.output_dir, 'global_step.bin'))
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
            logger.info("Load from output_dir {}".format(output_dir))

            optimizer.load_state_dict(torch.load(os.path.join(output_dir, 'optimizer.bin')))
            scheduler.load_state_dict(torch.load(os.path.join(output_dir, 'scheduler.bin')))
            # args = torch.load(os.path.join(output_dir, 'training_args.bin'))
            if hasattr(pretrained_model,'module'):
                pretrained_model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_pretrained_model.bin')))
            else: # Take care of distributed/parallel training
                pretrained_model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_pretrained_model.bin')))
            if hasattr(figer_model,'module'):
                figer_model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_task_model.bin')))
            else:
                figer_model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_task_model.bin')))


            global_step += 1
            start_epoch = int(global_step / len(train_dataloader))
            start_step = global_step-start_epoch*len(train_dataloader)-1
            logger.info("Start from global_step={} epoch={} step={}".format(global_step, start_epoch, start_step))
            # if args.local_rank in [-1, 0]:
            #     tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name, purge_step=global_step)

        else:
            global_step = 0
            start_epoch = 0
            start_step = 0
            # if args.local_rank in [-1, 0]:
            #     tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name, purge_step=global_step)

            logger.info("Start from scratch")
    else:
        global_step = 0
        start_epoch = 0
        start_step = 0
        logger.info("Start from scratch")
    tr_loss, logging_loss = 0.0, 0.0
    # model.zero_grad()
    pretrained_model.zero_grad()
    figer_model.zero_grad()
    # train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in range(start_epoch, int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            start = time.time()
            if args.restore and (step < start_step):
                continue

            # model.train()
            if args.freeze_bert:
                pretrained_model.eval()
            else:
                pretrained_model.train()
            figer_model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3],
                      'start_id':          batch[4],}
            # outputs = model(**inputs)
            pretrained_model_outputs = pretrained_model(**inputs)
            outputs = figer_model(pretrained_model_outputs,**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # epoch_iterator.set_description("loss {}".format(loss))
            logger.info("Epoch {}/{} - Iter {} / {}, loss = {:.5f}, time used = {:.3f}s".format(epoch, int(args.num_train_epochs),step,
                                                                                             len(train_dataloader),
                                                                                             loss.item(),
                                                                                             time.time() - start))
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(figer_model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(pretrained_model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                # model.zero_grad()
                pretrained_model.zero_grad()
                figer_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = figer_model.module if hasattr(figer_model, 'module') else figer_model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    model_to_save = pretrained_model.module if hasattr(pretrained_model, 'module') else pretrained_model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.bin'))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.bin'))
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    torch.save(global_step, os.path.join(args.output_dir, 'global_step.bin'))

                    logger.info("Saving model checkpoint, optimizer, global_step to %s", output_dir)
                    if (global_step/args.save_steps) > args.max_save_checkpoints:
                        try:
                            shutil.rmtree(os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step-args.max_save_checkpoints*args.save_steps)))
                        except OSError as e:
                            print(e)

                if args.local_rank == -1 and args.evaluate_during_training and global_step %args.eval_steps== 0:  # Only evaluate when single GPU otherwise metrics may not average well
                    model = (pretrained_model, figer_model)
                    results = evaluate(args, model, tokenizer)

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            break
        results = evaluate(args, model, tokenizer, prefix="")


    return global_step, tr_loss / global_step
def accuracy(out, l):
    cnt = 0
    y1 = []
    y2 = []
    for x1, x2 in zip(out, l):
        yy1 = []
        yy2 = []
        top = max(x1)
        for i in range(len(x1)):
            if x1[i] > 0 or x1[i] == top:
                yy1.append(i)
            if x2[i] > 0:
                yy2.append(i)
        y1.append(yy1)
        y2.append(yy2)
        cnt += set(yy1) == set(yy2)
    return cnt, y1, y2
save_results=[]
def evaluate(args, model, tokenizer, prefix=""):
    pretrained_model = model[0]
    figer_model = model[1]
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for dataset_type in ['test']:

        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):

            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, dataset_type, evaluate=True)

            if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # Eval!
            logger.info("***** Running evaluation {} *****".format(prefix))
            logger.info("  Num examples = %d", len(eval_dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)
            eval_loss = 0.0
            eval_accuracy = 0
            erine_eval_accuracy = 0
            my_accuracy = 0
            nb_eval_steps = 0
            nb_eval_examples = 0
            preds = None
            out_label_ids = None
            pred = []
            true = []
            erine_pred = []
            erine_true = []
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                pretrained_model.eval()
                figer_model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {'input_ids':      batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                              'labels':         batch[3],
                              'start_id':          batch[4]}
                    # outputs = model(**inputs)
                    pretrained_model_outputs = pretrained_model(**inputs)
                    outputs = figer_model(pretrained_model_outputs,**inputs)
                    tmp_eval_loss, logits = outputs[:2]
                    eval_loss += tmp_eval_loss.mean().item()
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                erine_preds = logits.detach().cpu().numpy()
                erine_out_label_ids = inputs['labels'].detach().cpu().numpy()
                tmp_eval_accuracy, tmp_pred, tmp_true = accuracy(preds, out_label_ids)
                erine_tmp_eval_accuracy, erine_tmp_pred, erine_tmp_true = accuracy(erine_preds, erine_out_label_ids)

                pred.extend(tmp_pred)
                true.extend(tmp_true)
                eval_accuracy += tmp_eval_accuracy

                erine_pred.extend(erine_pred)
                erine_true.extend(erine_true)
                erine_eval_accuracy += erine_tmp_eval_accuracy

                my_tmp = compute_metrics(eval_task,preds,out_label_ids)
                my_cnt = my_tmp[0]
                my_accuracy +=my_cnt
                nb_eval_examples += inputs['input_ids'].size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps

            erine_eval_accuracy = erine_eval_accuracy / nb_eval_examples

            def f1(p, r):
                if r == 0.:
                    return 0.
                return 2 * p * r / float(p + r)

            def loose_macro(true, pred):
                num_entities = len(true)
                p = 0.
                r = 0.
                for true_labels, predicted_labels in zip(true, pred):
                    if len(predicted_labels) > 0:
                        p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
                    if len(true_labels):
                        r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
                precision = p / num_entities
                recall = r / num_entities
                return precision, recall, f1(precision, recall)

            def loose_micro(true, pred):
                num_predicted_labels = 0.
                num_true_labels = 0.
                num_correct_labels = 0.
                for true_labels, predicted_labels in zip(true, pred):
                    num_predicted_labels += len(predicted_labels)
                    num_true_labels += len(true_labels)
                    num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
                if num_predicted_labels > 0:
                    precision = num_correct_labels / num_predicted_labels
                else:
                    precision = 0.
                recall = num_correct_labels / num_true_labels
                return precision, recall, f1(precision, recall)

            result = compute_metrics(eval_task, preds, out_label_ids)
            result = {'eval_loss': eval_loss,
                      'ERINE_accuracy':erine_eval_accuracy,
                      'ERINE_macro':loose_macro(true,pred),
                      'ERINE_micro':loose_micro(true,pred)
                    }

            logger.info('{} result: '.format(dataset_type))  #cnt, acc, loose_micro(y2, y1), loose_macro(y2, y1)
            logger.info('result:{}'.format(result))  #cnt, simple acc, loose_micro(y2, y1), loose_macro(y2, y1)

            results[dataset_type] = result
            save_result = str(results)

            save_results.append(save_result)
            if os.path.exists(os.path.join(args.output_dir, args.my_model_name + '_result.txt')):
                result_file = open(os.path.join(args.output_dir, args.my_model_name + '_result.txt'), 'a')
                # for line in save_results:
                result_file.write(str(results) + '\n')
                result_file.close()
            else:
                result_file = open(os.path.join(args.output_dir, args.my_model_name + '_result.txt'), 'w')
                for line in save_results:
                    result_file.write(str(line) + '\n')
                result_file.close()
    return results

def load_and_cache_examples(args, task, tokenizer, dataset_type, evaluate=False):
    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        dataset_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = processor.get_dev_examples(args.data_dir, dataset_type) if evaluate else processor.get_train_examples(args.data_dir, dataset_type)

        features = convert_examples_to_features_entity_typing(examples, args.label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    all_start_ids = torch.tensor([f.start_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_start_ids)
    return dataset

from pytorch_transformers.modeling_bert import BertEncoder
class Adapter(nn.Module):
    def __init__(self, args,adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config
        self.args = args
        self.down_project = nn.Linear(
            self.adapter_config.project_hidden_size,
            self.adapter_config.adapter_size,
        )
        self.encoder = BertEncoder(self.adapter_config)
        self.up_project = nn.Linear(self.adapter_config.adapter_size, adapter_config.project_hidden_size)

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)

        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=self.args.device)
        encoder_attention_mask = torch.ones(input_shape, device=self.args.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        head_mask = [None] * self.adapter_config.num_hidden_layers
        encoder_outputs = self.encoder(down_projected,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask)

        up_projected = self.up_project(encoder_outputs[0])
        return hidden_states + up_projected

class PretrainedModel(nn.Module):
    def __init__(self, args,num_labels):
        super(PretrainedModel, self).__init__()
        self.model = RobertaModel.from_pretrained("roberta-large", output_hidden_states=True)
        self.config = self.model.config
        self.config.freeze_adapter = args.freeze_adapter
        self.config.num_labels = num_labels
        if args.freeze_bert:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, start_id=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask)

        return outputs  # (loss), logits, (hidden_states), (attentions)
    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory), "Saving path should be a directory where the model and configuration can be saved"

        model_to_save = self.module if hasattr(self, 'module') else self
        model_to_save.config.save_pretrained(save_directory)
        output_model_file = os.path.join(save_directory, "pytorch_pretrained_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving pretrained model checkpoint to %s", save_directory)

class AdapterModel(nn.Module):
    def __init__(self, args, pretrained_model_config):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args
        self.adapter_size = args.adapter_size

        class AdapterConfig:
            project_hidden_size: int = self.config.hidden_size
            hidden_act: str = "gelu"
            adapter_size: int = self.adapter_size  # 64
            adapter_initializer_range: float = 0.0002
            is_decoder: bool = False
            attention_probs_dropout_prob: float= 0.1
            hidden_dropout_prob: float=0.1
            hidden_size: int=768
            initializer_range: float=0.02
            intermediate_size: int=3072
            layer_norm_eps: float=1e-05
            max_position_embeddings: int=514
            num_attention_heads: int=12
            num_hidden_layers: int=self.args.adapter_transformer_layers
            num_labels: int=2
            output_attentions: bool=False
            output_hidden_states: bool=False
            torchscript: bool=False
            type_vocab_size: int=1
            vocab_size: int=50265

        self.adapter_config = AdapterConfig

        self.adapter_skip_layers = self.args.adapter_skip_layers
        self.adapter_list = args.adapter_list
        self.adapter_num = len(self.adapter_list)
        self.adapter = nn.ModuleList([Adapter(args, AdapterConfig) for _ in range(self.adapter_num)])

    def forward(self, pretrained_model_outputs):
        outputs = pretrained_model_outputs
        sequence_output = outputs[0]
        # pooler_output = outputs[1]
        hidden_states = outputs[2]
        num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to('cuda')

        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapter):
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)
            adapter_hidden_states_count += 1
            if self.adapter_skip_layers >= 1:
                if adapter_hidden_states_count % self.adapter_skip_layers == 0:
                    hidden_states_last = hidden_states_last + adapter_hidden_states[int(adapter_hidden_states_count/self.adapter_skip_layers)]
        outputs = (hidden_states_last,) + outputs[2:]
        return outputs  # (loss), logits, (hidden_states), (attentions)

class FIGERModel(nn.Module):
    def __init__(self, args, pretrained_model_config, fac_adapter,et_adapter,lin_adapter):
        super(FIGERModel, self).__init__()

        self.args = args
        self.config = pretrained_model_config
        self.num_labels = self.config.num_labels

        self.fac_adapter = fac_adapter
        self.et_adapter = et_adapter
        self.lin_adapter = lin_adapter

        if args.freeze_adapter and (self.fac_adapter is not None):
            for p in self.fac_adapter.parameters():
                p.requires_grad = False
        if args.freeze_adapter and (self.et_adapter is not None):
            for p in self.et_adapter.parameters():
                p.requires_grad = False
        if args.freeze_adapter and (self.lin_adapter is not None):
            for p in self.lin_adapter.parameters():
                p.requires_grad = False
        self.adapter_num = 0
        if self.fac_adapter is not None:
            self.adapter_num += 1
        if self.et_adapter is not None:
            self.adapter_num += 1
        if self.lin_adapter is not None:
            self.adapter_num += 1

        if self.args.task_adapter:
            self.adapter_num += 1
            self.task_adapter = AdapterModel(self.args, pretrained_model_config)

        if self.args.fusion_mode == 'concat':
            self.task_dense_lin = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
            self.task_dense_fac = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
            self.task_dense = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.config.hidden_size, self.num_labels)


    def forward(self,pretrained_model_outputs,input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, start_id=None):

        batch = input_ids.size(0)
        # print('input_ids:{}'.format(input_ids))
        # print('attention_mask:{}'.format(attention_mask))
        pretrained_model_last_hidden_states = pretrained_model_outputs[0]
        if self.fac_adapter is not None:
            fac_adapter_outputs, _ = self.fac_adapter(pretrained_model_outputs)
        if self.et_adapter is not None:
            et_adapter_outputs, _ = self.et_adapter(pretrained_model_outputs)
        if self.lin_adapter is not None:
            lin_adapter_outputs, _ = self.lin_adapter(pretrained_model_outputs)
        if self.args.task_adapter:
            task_adapter_outputs, _ = self.task_adapter(pretrained_model_outputs)

        if self.args.fusion_mode == 'add':
            task_features = pretrained_model_last_hidden_states
            if self.args.task_adapter:
                task_features = task_features + task_adapter_outputs
            if self.fac_adapter is not None:
                task_features = task_features + fac_adapter_outputs
            if self.et_adapter is not None:
                task_features = task_features + et_adapter_outputs
            if self.lin_adapter is not None:
                task_features = task_features + lin_adapter_outputs
        elif self.args.fusion_mode == 'concat':
            combine_features = pretrained_model_last_hidden_states
            fac_features = self.task_dense_fac(torch.cat([combine_features, fac_adapter_outputs], dim=2))
            lin_features = self.task_dense_lin(torch.cat([combine_features, lin_adapter_outputs], dim=2))
            task_features = self.task_dense(torch.cat([fac_features, lin_features], dim=2))

        start_id = start_id.unsqueeze(1)
        entity_output = torch.bmm(start_id, task_features)
        entity_output = entity_output.squeeze(1)
        logits = self.out_proj(self.dropout(self.dense(entity_output)))

        outputs = (logits,) + pretrained_model_outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # loss_fct = CrossEntropyLoss()
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_task_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving task model checkpoint to %s", save_directory)


def load_pretrained_adapter(adapter, adapter_path):
    new_adapter= adapter
    model_dict = new_adapter.state_dict()
    logger.info('Adapter model weight:')
    logger.info(new_adapter.state_dict().keys())
    # print(model_dict['bert.encoder.layer.2.intermediate.dense.weight'])
    logger.info('Load model state dict from {}'.format(adapter_path))
    adapter_meta_dict = torch.load(adapter_path, map_location=lambda storage, loc: storage)
    for item in ['out_proj.bias', 'out_proj.weight', 'dense.weight',
                 'dense.bias']:  # 'adapter.down_project.weight','adapter.down_project.bias','adapter.up_project.weight','adapter.up_project.bias'
        if item in adapter_meta_dict:
            adapter_meta_dict.pop(item)
    changed_adapter_meta = {}
    for key in adapter_meta_dict.keys():
        changed_adapter_meta[key.replace('adapter.', 'adapter.')] = adapter_meta_dict[key]
    changed_adapter_meta = {k: v for k, v in changed_adapter_meta.items() if k in model_dict.keys()}
    model_dict.update(changed_adapter_meta)
    new_adapter.load_state_dict(model_dict)
    return new_adapter

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--comment", default='', type=str,
                        help="The comment")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--freeze_bert", default=True, type=bool,
                        help="freeze the parameters of pretrained model.")
    parser.add_argument("--freeze_adapter", default=False, type=bool,
                        help="freeze the parameters of adapter.")
    parser.add_argument("--test_mode", default=0, type=int,
                        help="test freeze adapter")

    parser.add_argument('--fusion_mode', type=str, default='concat',help='the fusion mode for bert feautre and adapter feature |add|concat')
    parser.add_argument("--adapter_transformer_layers", default=2, type=int,
                        help="The transformer layers of adapter.")
    parser.add_argument("--adapter_size", default=768, type=int,
                        help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,11,22", type=str,
                        help="The layer where add an adapter")
    parser.add_argument("--adapter_skip_layers", default=3, type=int,
                        help="The skip_layers of adapter according to bert layers")

    parser.add_argument('--task_adapter', default='',type=str, help='Whether to use task adapter')
    parser.add_argument('--meta_fac_adaptermodel', default='',type=str, help='the pretrained factual adapter model')
    parser.add_argument('--meta_et_adaptermodel', default='',type=str, help='the pretrained entity typing adapter model')
    parser.add_argument('--meta_lin_adaptermodel', default='', type=str, help='the pretrained linguistic adapter model')
    parser.add_argument('--meta_bertmodel', default='', type=str, help='the pretrained bert model')

    parser.add_argument("--restore", type=bool, default=True, help="Whether restore from the last checkpoint, is nochenckpoints, start from scartch")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--eval_steps', type=int, default=None,
                        help="eval every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--max_save_checkpoints', type=int, default=500,
                        help="The max amounts of checkpoint saving. Bigger than it will delete the former checkpoints")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    args.adapter_list = args.adapter_list.split(',')
    args.adapter_list = [int(i) for i in args.adapter_list]

    name_prefix = 'batch-'+str(args.per_gpu_train_batch_size)+'_'+'lr-'+str(args.learning_rate)+'_'+'warmup-'+str(args.warmup_steps)+'_'+'epoch-'+str(args.num_train_epochs)+'_'+str(args.comment)
    args.my_model_name = args.task_name+'_'+name_prefix
    args.output_dir = os.path.join(args.output_dir, args.my_model_name)

    if args.eval_steps is None:
        args.eval_steps = args.save_steps*1
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #     raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    if not os.path.exists(os.path.join(args.data_dir,'labels.json')):
        logger.info("label_list not exist, creating.....")
        with open(os.path.join(args.data_dir,'train.json'), 'r', encoding='utf8') as f:
            examples = []
            lines = json.load(f)
            for (i, line) in enumerate(lines):
                label = line['labels']
                examples.append(label)
            d = {}
            for e in examples:
                for l in e:
                    if l in d:
                        d[l] += 1
                    else:
                        d[l] = 1
            for k, v in d.items():
                d[k] = (len(examples) - v) * 1. / v

            label_list = list(d.keys())
            examples = []
        with open(os.path.join(args.data_dir,'labels.json'), "w") as f:
            json.dump(label_list, f)
    else:
        logger.info("label_list exist, loading.....")
        with open(os.path.join(args.data_dir,'labels.json'), "r") as f:
            label_list = json.load(f)
        # label_list = json.load(os.path.join(args.data_dir,'labels.json'))


    # label_list = processor.get_labels()
    num_labels = len(label_list)
    args.label_list = label_list

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    # model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    pretrained_model = PretrainedModel(args, num_labels)
    if args.meta_fac_adaptermodel:
        fac_adapter = AdapterModel(args, pretrained_model.config)
        fac_adapter = load_pretrained_adapter(fac_adapter,args.meta_fac_adaptermodel)
    else:
        fac_adapter = None
    if args.meta_et_adaptermodel:
        et_adapter = AdapterModel(args, pretrained_model.config)
        et_adapter = load_pretrained_adapter(et_adapter,args.meta_et_adaptermodel)
    else:
        et_adapter = None
    if args.meta_lin_adaptermodel:
        lin_adapter = AdapterModel(args, pretrained_model.config)
        lin_adapter = load_pretrained_adapter(lin_adapter,args.meta_lin_adaptermodel)
    else:
        lin_adapter = None
    # adapter_model = AdapterModel(pretrained_model.config,num_labels,args.adapter_size,args.adapter_interval,args.adapter_skip_layers)

    figer_model = FIGERModel(args, pretrained_model.config, fac_adapter=fac_adapter, et_adapter=et_adapter, lin_adapter=lin_adapter)

    if args.meta_bertmodel:
        model_dict = pretrained_model.state_dict()
        bert_meta_dict = torch.load(args.meta_bertmodel, map_location=lambda storage, loc: storage)
        for item in ['out_proj.weight', 'out_proj.bias', 'dense.weight', 'dense.bias', 'lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias',
                     'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight']:
            if item in bert_meta_dict:
                bert_meta_dict.pop(item)

        changed_bert_meta = {}
        for key in bert_meta_dict.keys():
            changed_bert_meta[key.replace('model.','roberta.')] = bert_meta_dict[key]
        changed_bert_meta = {k: v for k, v in changed_bert_meta.items() if k in model_dict.keys()}
        model_dict.update(changed_bert_meta)
        pretrained_model.load_state_dict(model_dict)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    pretrained_model.to(args.device)
    figer_model.to(args.device)

    model = (pretrained_model, figer_model)

    logger.info("Training/evaluation parameters %s", args)
    # val_dataset = load_and_cache_examples(args, args.task_name, tokenizer, 'dev', evaluate=True)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, dataset_type='train', evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = figer_model.module if hasattr(figer_model,
                                                      'module') else figer_model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        model_to_save = pretrained_model.module if hasattr(pretrained_model,
                                                           'module') else pretrained_model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         # model = model_class.from_pretrained(checkpoint)
    #         # model.to(args.device)
    #         result = evaluate(args, model, tokenizer, prefix=global_step)
    #         result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #         results.update(result)
    # save_result = str(results)
    #
    # save_results.append(save_result)
    # result_file = open(os.path.join(args.output_dir, args.my_model_name + '_result.txt'), 'w')
    # for line in save_results:
    #     result_file.write(str(line) + '\n')
    # result_file.close()
    return results


if __name__ == "__main__":
    main()
