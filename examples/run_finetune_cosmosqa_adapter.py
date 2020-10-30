# coding=utf-8
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
""" k-adapter for CosmosQA"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import json

import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from pytorch_transformers.my_modeling_roberta import RobertaConfig, RoBERTaForMultipleChoice, \
    RobertaForSequenceClassification
from pytorch_transformers import AdamW, WarmupLinearSchedule,RobertaModel

from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from itertools import cycle
import time
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RoBERTaForMultipleChoice, RobertaTokenizer),
}
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (RobertaConfig,)), ())

logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 idx,
                 context_sentence,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label=None):
        self.idx = idx
        self.context_sentence = context_sentence
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "id: {}".format(self.idx),
            "context_sentence: {}".format(self.context_sentence),
            "ending_0: {}".format(self.endings[0]),
            "ending_1: {}".format(self.endings[1]),
            "ending_2: {}".format(self.endings[2]),
            "ending_3: {}".format(self.endings[3]),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return "\n".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def read_examples_origin(input_file, is_training):
    cont = 0
    examples = []
    with open(input_file) as f:
        for line in f:
            line = json.loads(line.strip())

            context = line['context']
            question = line['question']
            qa_list = [line['answer0'], line['answer1'], line['answer2'], line['answer3']]

            examples.append(
                Example(
                    idx=cont,
                    context_sentence=context,
                    ending_0=question + ' </s> '  + qa_list[0],
                    ending_1=question + ' </s> '  + qa_list[1],
                    ending_2=question + ' </s> '  + qa_list[2],
                    ending_3=question + ' </s> '  + qa_list[3],
                    label=line['label'] if is_training else None
                )
            )
    return examples

def read_examples_add_evidence(input_file, is_training):
    cont = 0
    examples = []
    with open(input_file) as f:
        for line in f:
            line = json.loads(line.strip())

            context = line['context']
            question = line['question']
            qa_list = [line['answer0'], line['answer1'], line['answer2'], line['answer3']]
            evidencen_list = [line['answer0_searched_evidence']['basic'], line['answer1_searched_evidence']['basic'],
                             line['answer2_searched_evidence']['basic'], line['answer3_searched_evidence']['basic']]
            evidence_text = []
            for evidence in evidencen_list:
                tmp = []
                for item in evidence:
                    tmp.append(item['text'])
                evidence_text.append(' # '.join(tmp))

            examples.append(
                Example(
                    idx=cont,
                    context_sentence=context,
                    ending_0=  question + ' </s> '  + qa_list[0] + ' </s> ' + evidence_text[0],
                    ending_1=  question + ' </s> '  + qa_list[1] + ' </s> ' + evidence_text[1],
                    ending_2=  question + ' </s> '  + qa_list[2] + ' </s> ' + evidence_text[2],
                    ending_3=  question + ' </s> '  + qa_list[3] + ' </s> ' + evidence_text[3],
                    label=line['label'] if is_training else None
                )
            )
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.

    features = []
    for example_index, example in enumerate(examples):
        if example_index % 1000 == 0:
            logger.info('Processing {} examples...'.format(example_index))
        context_tokens = tokenizer.tokenize(example.context_sentence)
        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            # ending_tokens =tokenizer.tokenize('The answer is')+ tokenizer.tokenize(ending)
            ending_tokens = tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ['<s>'] + context_tokens_choice + ["</s>"] + ending_tokens + ["</s>"]
            # segment_ids = [0] * (len(context_tokens_choice) + 1) + [1] * (len(ending_tokens) + 1)+[2]
            # segment_ids = [0] * (len(context_tokens_choice) + 1) + [0] * (len(ending_tokens) + 1) + [0]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids =  input_ids + ([0] * padding_length)

            input_mask =  input_mask + ([0] * padding_length)

            # segment_ids = ([4] * padding_length) + segment_ids
            segment_ids =  segment_ids + ([0] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = int(example.label) if example.label else None
        if example_index < 1 and is_training:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example.idx))
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                logger.info("tokens:{}".format(' '.join(tokens)))
                logger.info("choice: {}".format(choice_idx))
                # logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.idx,
                choices_features=choices_features,
                label=label
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
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

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-self.adapter_config.initializer_range, self.adapter_config.initializer_range)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-self.adapter_config.initializer_range, self.adapter_config.initializer_range)


class PretrainedModel(nn.Module):
    def __init__(self, args):
        super(PretrainedModel, self).__init__()
        self.model = RobertaModel.from_pretrained("roberta-large", num_labels=5, output_hidden_states=True)

        self.config = self.model.config
        self.config.freeze_adapter = args.freeze_adapter
        if args.freeze_bert:
            for p in self.parameters():
                p.requires_grad = False


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.model(input_ids=flat_input_ids,
                             position_ids=flat_position_ids,
                             token_type_ids=flat_token_type_ids,
                             attention_mask=flat_attention_mask,
                             head_mask=head_mask)

        return outputs  # (loss), logits, (hidden_states), (attentions)
    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_pretrained_model.bin")

        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)


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

        if self.args.fusion_mode == 'concat':
            self.task_dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        if args.test_mode == 2:
            self.init_weights(0.1)
    def init_weights(self, init_range):
        logging.info("Initialize the parameters of the taskdense")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-init_range, init_range)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-init_range, init_range)

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

        if self.args.fusion_mode == 'add':
            task_features = self.args.a_rate * sequence_output+self.args.b_rate * hidden_states_last
        elif self.args.fusion_mode == 'concat':
            task_features = self.task_dense(torch.cat([self.args.a_rate * sequence_output, self.args.b_rate * hidden_states_last],dim=2))
            # task_features = self.dropout(task_features)

        outputs = (task_features,) + outputs[2:]

        return outputs  # (loss), logits, (hidden_states), (attentions)


class COSMOSQAModel(nn.Module):
    def __init__(self, args, pretrained_model_config, fac_adapter, et_adapter, lin_adapter):
        super(COSMOSQAModel, self).__init__()
        self.args = args
        self.config = pretrained_model_config

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

        if self.args.fusion_mode == 'concat':
            self.task_dense_lin = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
            self.task_dense_fac = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
            self.task_dense = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)

        # self.num_labels = config.num_labels
        self.config = pretrained_model_config
        self.num_labels = 4
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(self, pretrained_model_outputs, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        pretrained_model_last_hidden_states = pretrained_model_outputs[0]
        if self.fac_adapter is not None:
            fac_adapter_outputs, _ = self.fac_adapter(pretrained_model_outputs)
        if self.et_adapter is not None:
            et_adapter_outputs, _ = self.et_adapter(pretrained_model_outputs)
        if self.lin_adapter is not None:
            lin_adapter_outputs, _ = self.lin_adapter(pretrained_model_outputs)

        if self.args.fusion_mode == 'add':
            task_features = pretrained_model_last_hidden_states
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

        sequence_output = self.dropout(task_features)
        logits = self.classifier(sequence_output[:, 0, :].squeeze(dim=1))
        reshaped_logits = logits.view(-1, self.num_labels)

        outputs = (reshaped_logits,) + pretrained_model_outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = loss
        else:
            outputs = outputs[0]

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)

def load_pretrained_adapter(adapter, adapter_path):
    new_adapter= adapter
    model_dict = new_adapter.state_dict()
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
    parser.add_argument("--task_name", default='cosmosqa', type=str,
                        help="The name of the task to train selected in the list")
    parser.add_argument("--comment", default='', type=str,
                        help="The comment")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

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

    parser.add_argument('--meta_fac_adaptermodel', default='',type=str, help='the pretrained factual adapter model')
    parser.add_argument('--meta_et_adaptermodel', default='',type=str, help='the pretrained entity typing adapter model')
    parser.add_argument('--meta_lin_adaptermodel', default='', type=str, help='the pretrained linguistic adapter model')

    parser.add_argument('--meta_bertmodel', default='', type=str, help='the pretrained bert model')

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
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to make predictions on the test set")
    parser.add_argument("--print_error_analysis", action='store_true',
                        help='print the errors in valid dataset')
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

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
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--report_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")


    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
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
    parser.add_argument('--preprocess_type', type=str, default='', help="How to process the input")
    args = parser.parse_args()
    args = parser.parse_args()

    args.adapter_list = args.adapter_list.split(',')
    args.adapter_list = [int(i) for i in args.adapter_list]

    name_prefix = 'batch-'+str(args.per_gpu_train_batch_size)+'_'+'lr-'+str(args.learning_rate)+'_'+'warmup-'+str(args.warmup_steps)+'_'+'epoch-'+str(args.num_train_epochs)+'_'+str(args.comment)
    args.my_model_name = args.task_name+'_'+name_prefix
    args.output_dir = os.path.join(args.output_dir, args.my_model_name)

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

    try:
        os.makedirs(args.output_dir)
    except:
        pass

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    pretrained_model = PretrainedModel(args)
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
    cosmosqa_model = COSMOSQAModel(args,pretrained_model.config,fac_adapter=fac_adapter, et_adapter=et_adapter, lin_adapter=lin_adapter)

    if args.meta_bertmodel:
        model_dict = pretrained_model.state_dict()
        logger.info('Roberta model roberta.embeddings.word_embeddings.weight:')
        logger.info(pretrained_model.state_dict()['roberta.embeddings.word_embeddings.weight'])
        # print(model_dict['bert.encoder.layer.2.intermediate.dense.weight'])
        logger.info('Load pertrained bert model state dict from {}'.format(args.meta_bertmodel))
        bert_meta_dict = torch.load(args.meta_bertmodel, map_location=lambda storage, loc: storage)
        # print(model.state_dict().keys())
        # print(bert_meta_dict.keys())
        # if 'out_proj.weight' in bert_meta_dict:
        #     bert_meta_dict.pop('out_proj.weight')
        # if 'out_proj.bias' in bert_meta_dict:
        #     bert_meta_dict.pop('out_proj.bias')
        for item in ['out_proj.weight', 'out_proj.bias', 'dense.weight', 'dense.bias', 'lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias',
                     'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight']:
            if item in bert_meta_dict:
                bert_meta_dict.pop(item)

        changed_bert_meta = {}
        for key in bert_meta_dict.keys():
            changed_bert_meta[key.replace('model.','roberta.')] = bert_meta_dict[key]
        # print(changed_bert_meta.keys())
        changed_bert_meta = {k: v for k, v in changed_bert_meta.items() if k in model_dict.keys()}
        # print(changed_bert_meta.keys())
        model_dict.update(changed_bert_meta)
        pretrained_model.load_state_dict(model_dict)
        logger.info('RoBERTa-meta new model roberta.embeddings.word_embeddings.weight:')
        # logger.info(model.state_dict()['bert.embeddings.word_embeddings.weight'])
        logger.info(pretrained_model.state_dict()['roberta.embeddings.word_embeddings.weight'])

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    pretrained_model.to(args.device)
    cosmosqa_model.to(args.device)
    # model.to(args.device)
    model = (pretrained_model, cosmosqa_model)

    logger.info("Training/evaluation parameters %s", args)


    # if args.fp16:
    #     model.half()
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        if args.freeze_bert:
            cosmosqa_model = torch.nn.DataParallel(cosmosqa_model)
        else:
            pretrained_model = torch.nn.DataParallel(pretrained_model)
            cosmosqa_model = torch.nn.DataParallel(cosmosqa_model)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    read_examples_dict = {
        'read_examples_origin': read_examples_origin,
        'read_examples_add_evidence': read_examples_add_evidence
    }
    convert_examples_to_features_dict = {
        'read_examples_origin': convert_examples_to_features,
        'read_examples_add_evidence': convert_examples_to_features,
    }

    if args.do_train:
        # Prepare data loader
        # tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name)
        train_examples = read_examples_dict[args.preprocess_type](os.path.join(args.data_dir, 'train.jsonl'),
                                                                  is_training=True)
        train_features = convert_examples_to_features_dict[args.preprocess_type](
            train_examples, tokenizer, args.max_seq_length, True)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        num_train_optimization_steps = args.train_steps

        # Prepare optimizer
        #
        # param_optimizer = list(model.named_parameters())
        #
        # # hack to remove pooler, which is not used
        # # thus it produce None grad that break apex
        # param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        if args.freeze_bert:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in cosmosqa_model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in cosmosqa_model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in cosmosqa_model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in cosmosqa_model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0},
                {'params': [p for n, p in pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in pretrained_model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                         t_total=num_train_optimization_steps)

        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_acc = 0
        if args.freeze_bert:
            pretrained_model.eval()
        else:
            pretrained_model.train()
        cosmosqa_model.train()
        tr_loss, logging_loss = 0.0, 0.0
        nb_tr_examples, nb_tr_steps = 0, 0
        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)
        eval_flag = True
        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            # loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            pretrained_model_outputs = pretrained_model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask, labels=label_ids)
            outputs = cosmosqa_model(pretrained_model_outputs,input_ids=input_ids, token_type_ids=None, attention_mask=input_mask, labels=label_ids)

            loss = outputs  # model outputs are always tuple in pytorch-transformers (see doc)

            # loss = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask, labels=label_ids)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.fp16 and args.loss_scale != 1.0:
                loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # print(loss)
            tr_loss += loss.item()
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                eval_flag = True

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        # results = evaluate(args, model, tokenizer)
                        # for key, value in results.items():
                        #     tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    # tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

            # if (global_step + 1) %args.report_steps==0:
            #     tr_loss = 0
            #     nb_tr_examples, nb_tr_steps = 0, 0
            #     logger.info("***** Report result *****")
            #     logger.info("  %s = %s", 'global_step', str(global_step+1))
            #     logger.info("  %s = %s", 'train loss', str(train_loss))
            print('global_step:',global_step)
            print('args.eval_steps:',args.eval_steps)
            print(args.do_eval)

            print((global_step + 1) % args.eval_steps == 0)
            print(eval_flag)
            print(args.do_eval and ((global_step + 1) % args.eval_steps == 0) and eval_flag)
            # step % args.gradient_accumulation_steps == 1
            if args.do_eval and ((global_step + 1) % args.eval_steps == 0) and eval_flag:
                eval_flag = False
                print('eval...')
                for file in ['valid.jsonl']:
                    eval_examples = read_examples_dict[args.preprocess_type](os.path.join(args.data_dir, file),
                                                                             is_training=True)
                    inference_labels = []
                    gold_labels = []
                    eval_features = convert_examples_to_features_dict[args.preprocess_type](
                        eval_examples, tokenizer, args.max_seq_length, False)
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)
                    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
                    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
                    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
                    # Run prediction for full data
                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    pretrained_model.eval()
                    cosmosqa_model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                        start = time.time()
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)

                        with torch.no_grad():
                            # tmp_eval_loss= model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
                            # logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                            # tmp_eval_loss = model(input_ids=input_ids, token_type_ids=None,
                            #                       attention_mask=input_mask, labels=label_ids)
                            # logits = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)
                            pretrained_model_outputs = pretrained_model(input_ids=input_ids, token_type_ids=None,
                                                  attention_mask=input_mask, labels=label_ids)
                            tmp_eval_loss = cosmosqa_model(pretrained_model_outputs, input_ids=input_ids, token_type_ids=None, attention_mask=input_mask, labels=label_ids)
                            logits = cosmosqa_model(pretrained_model_outputs, input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        tmp_eval_accuracy = accuracy(logits, label_ids)
                        # if nb_eval_steps==0:
                        #    print(logits)
                        inference_labels.append(np.argmax(logits, axis=1))
                        gold_labels.append(label_ids)
                        eval_loss += tmp_eval_loss.mean().item()
                        eval_accuracy += tmp_eval_accuracy

                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                        logger.info(
                            "Validation Iter {} / {}, loss = {:.5f}, accuracy = {}, time used = {:.3f}s".format(nb_eval_steps,
                                                                                                 len(eval_dataloader),
                                                                                                 tmp_eval_loss.mean().item(),
                                                                                                tmp_eval_accuracy.item()/input_ids.size(0),
                                                                                                time.time() - start))

                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = eval_accuracy / nb_eval_examples

                    result = {'eval_loss': eval_loss,
                              'eval_accuracy': eval_accuracy,
                              'global_step': global_step + 1,
                              'loss': train_loss}

                    logger.info('result:{}'.format(result))

                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    with open(output_eval_file, "a", encoding='utf8') as writer:
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
                        writer.write('*' * 80)
                        writer.write('\n')

                    # for key, value in result.items():
                    #     tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                    model_to_save = cosmosqa_model.module if hasattr(cosmosqa_model, 'module') else cosmosqa_model  # Take care of distributed/parallel training
                    # model_to_save = model.module if hasattr(model,
                    #                                         'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir,
                                                     "pytorch_model_{}_{}.bin".format(global_step + 1,
                                                                                      eval_accuracy))
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save = pretrained_model.module if hasattr(pretrained_model, 'module') else pretrained_model
                    output_model_file = os.path.join(args.output_dir,
                                                     "pytorch_bertmodel_{}_{}.bin".format(global_step + 1,
                                                                                      eval_accuracy))
                    torch.save(model_to_save.state_dict(), output_model_file)

                    if eval_accuracy > best_acc and 'dev' in file:
                        print("=" * 80)
                        print("Best Acc", eval_accuracy)
                        print("Saving Model......")
                        best_acc = eval_accuracy
                        # Save a trained model
                        model_to_save = cosmosqa_model.module if hasattr(cosmosqa_model,
                                                                   'module') else cosmosqa_model  # Take care of distributed/parallel training
                        output_model_file = os.path.join(args.output_dir, "pytorch_model_best.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save = pretrained_model.module if hasattr(pretrained_model,
                                                                           'module') else pretrained_model
                        output_model_file = os.path.join(args.output_dir, "pytorch_bertmodel_best.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                    #     print("=" * 80)
                    #     inference_labels = np.concatenate(inference_labels, 0)
                    #     gold_labels = np.concatenate(gold_labels, 0)
                    #     with open(os.path.join(args.output_dir, "error_output.txt", ), 'w', encoding='utf8') as f:
                    #         for i in range(len(eval_examples)):
                    #             if inference_labels[i] != gold_labels[i]:
                    #                 f.write(str(repr(eval_examples[i])) + '\n')
                    #                 f.write(str(inference_labels[i]) + '\n')
                    #                 f.write("=" * 80 + '\n')
                    # else:
                    #     print("=" * 80)
                if args.freeze_bert:
                    pretrained_model.eval()
                else:
                    pretrained_model.train()
                cosmosqa_model.train()
                # model.train()

    #
    #
    # example_ids, contexts, questions, answers = [], [], [], []
    # if args.print_error_analysis:
    #     with open(os.path.join(args.data_dir, 'valid.jsonl'), 'r', encoding='utf8') as f:
    #         for line in f:
    #             line = json.loads(line.strip())
    #             example_ids.append(line['id'])
    #             contexts.append(line['context'])
    #             questions.append(line['question'])
    #             answers.append([line['answer0'], line['answer1'], line['answer2'], line['answer3']])
    #
    #     logger.info('Load model from roberta_cosmosqa_baseline/pytorch_model_800_0.8144053601340033.bin')
    #     trained_models = torch.load('roberta_cosmosqa_baseline/pytorch_model_800_0.8144053601340033.bin')
    #     changed_trained_models = {}
    #     for key in trained_models:
    #         changed_trained_models[key.replace('module.','')] = trained_models[key]
    #
    #     model.load_state_dict(changed_trained_models)
    #
    #     for file in ['valid.jsonl']:
    #         eval_examples = read_examples_dict[args.preprocess_type](os.path.join(args.data_dir, file),
    #                                                                  is_training=True)
    #         inference_labels = []
    #         gold_labels = []
    #         eval_features = convert_examples_to_features_dict[args.preprocess_type](
    #             eval_examples, tokenizer, args.max_seq_length, False)
    #         logger.info("***** Running evaluation *****")
    #         logger.info("  Num examples = %d", len(eval_examples))
    #         logger.info("  Batch size = %d", args.eval_batch_size)
    #         all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    #         all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    #         all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    #         all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
    #         eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    #         # Run prediction for full data
    #         eval_sampler = SequentialSampler(eval_data)
    #         eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    #
    #         model.eval()
    #         eval_loss, eval_accuracy = 0, 0
    #         nb_eval_steps, nb_eval_examples = 0, 0
    #         all_logits = []
    #         index = 0
    #         for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, total=2985):
    #             input_ids = input_ids.to(device)
    #             input_mask = input_mask.to(device)
    #             segment_ids = segment_ids.to(device)
    #             label_ids = label_ids.to(device)
    #             index+= 1
    #             # if index%100 == 0:
    #             #     logger.info('Evaluate {} examples ...'.format(index))
    #
    #             with torch.no_grad():
    #                 # tmp_eval_loss= model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
    #                 # logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
    #                 tmp_eval_loss = model(input_ids=input_ids, token_type_ids=None,
    #                                       attention_mask=input_mask, labels=label_ids)
    #                 logits = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)
    #             logits = logits.detach().cpu().numpy()
    #             label_ids = label_ids.to('cpu').numpy()
    #             tmp_eval_accuracy = accuracy(logits, label_ids)
    #             for i in range(logits.shape[0]):
    #                 all_logits.append(logits[i])
    #             inference_labels.append(np.argmax(logits, axis=1)[0])
    #             gold_labels.append(label_ids[0])
    #             eval_loss += tmp_eval_loss.mean().item()
    #             eval_accuracy += tmp_eval_accuracy
    #
    #             nb_eval_examples += input_ids.size(0)
    #             nb_eval_steps += 1
    #
    #
    #         eval_loss = eval_loss / nb_eval_steps
    #         eval_accuracy = eval_accuracy / nb_eval_examples
    #
    #         logger.info('Eval accuracy:{}'.format(eval_accuracy))
    #
    #         with open('data/valid_errors.txt','w',encoding='utf8') as f:
    #             error_number = 0
    #             for i in range(len(gold_labels)):
    #                 if inference_labels[i] != gold_labels[i]:
    #                     error_number += 1
    #                     f.write('id:{}\n'.format(example_ids[i]))
    #                     f.write('context:{}\n\n'.format(contexts[i]))
    #                     f.write('question:{}\n'.format(questions[i]))
    #                     f.write('answer 0:{}\n'.format(answers[i][0]))
    #                     f.write('answer 1:{}\n'.format(answers[i][1]))
    #                     f.write('answer 2:{}\n'.format(answers[i][2]))
    #                     f.write('answer 3:{}\n'.format(answers[i][3]))
    #                     f.write('right answer:{}, selected answer:{}\n'.format(gold_labels[i], inference_labels[i]))
    #                     f.write('right answer prob:{}, selected answer prob:{}\n'.format(all_logits[i][gold_labels[i]], all_logits[i][inference_labels[i]]))
    #                     f.write('\n'*5)
    #
    #             print('Error number:{}'.format(error_number))
    #
    #
    # if args.do_test:
    #     logger.info(os.path.join(args.output_dir, "pytorch_model_best.bin"))
    #     model.load_state_dict(torch.load(os.path.join(args.output_dir, "pytorch_model_best.bin")))
    #
    #     for file in ['test.jsonl']:
    #         eval_examples = read_examples_dict[args.preprocess_type](os.path.join(args.data_dir, file),
    #                                                                  is_training=False)
    #         inference_labels = []
    #         gold_labels = []
    #         eval_features = convert_examples_to_features_dict[args.preprocess_type](
    #             eval_examples, tokenizer, args.max_seq_length, False)
    #         logger.info("***** Running testing *****")
    #         logger.info("  Num examples = %d", len(eval_examples))
    #         logger.info("  Batch size = %d", args.eval_batch_size)
    #         all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    #         all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    #         all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    #         eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    #
    #         eval_sampler = SequentialSampler(eval_data)
    #         eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    #
    #         model.eval()
    #         all_logits = []
    #         for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, total=6963):
    #             input_ids = input_ids.to(device)
    #             input_mask = input_mask.to(device)
    #             segment_ids = segment_ids.to(device)
    #             with torch.no_grad():
    #                 logits = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)
    #             logits = logits.detach().cpu().numpy()
    #             for i in range(logits.shape[0]):
    #                 all_logits.append(logits[i])
    #             inference_labels.append(np.argmax(logits, axis=1)[0])
    #         with open(os.path.join(args.output_dir, "test_results.test"),'w',encoding='utf8') as f:
    #             f.write('\n'.join(map(str, inference_labels)))

    # if args.do_test:
    #     logger.info( os.path.join(args.output_dir, "pytorch_model_best.bin"))
    #     model.load_state_dict(torch.load( os.path.join(args.output_dir, "pytorch_model_best.bin")))
    #
    #     for file in ['test.jsonl']:
    #         eval_examples = read_examples_dict[args.preprocess_type](os.path.join(args.data_dir, file),
    #                                                                  is_training=False)
    #         inference_labels = []
    #         gold_labels = []
    #         eval_features = convert_examples_to_features_dict[args.preprocess_type](
    #             eval_examples, tokenizer, args.max_seq_length, False)
    #         logger.info("***** Running testing *****")
    #         logger.info("  Num examples = %d", len(eval_examples))
    #         logger.info("  Batch size = %d", args.eval_batch_size)
    #         all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    #         all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    #         all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    #         all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
    #         eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    #
    #         eval_sampler = SequentialSampler(eval_data)
    #         eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    #
    #         model.eval()
    #         eval_loss, eval_accuracy = 0, 0
    #         nb_eval_steps, nb_eval_examples = 0, 0
    #         all_logits = []
    #         for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
    #             input_ids = input_ids.to(device)
    #             input_mask = input_mask.to(device)
    #             segment_ids = segment_ids.to(device)
    #             label_ids = label_ids.to(device)
    #
    #             with torch.no_grad():
    #                 # tmp_eval_loss= model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
    #                 # logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
    #                 tmp_eval_loss = model(input_ids=input_ids, token_type_ids=None,
    #                                       attention_mask=input_mask, labels=label_ids)
    #                 logits = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)
    #             logits = logits.detach().cpu().numpy()
    #             label_ids = label_ids.to('cpu').numpy()
    #             tmp_eval_accuracy = accuracy(logits, label_ids)
    #             # if nb_eval_steps==0:
    #             #    print(logits)
    #             inference_labels.append(np.argmax(logits, axis=1))
    #             gold_labels.append(label_ids)
    #             eval_accuracy += tmp_eval_accuracy
    #
    #             nb_eval_examples += input_ids.size(0)
    #             nb_eval_steps += 1
    #
    #         eval_accuracy = eval_accuracy / nb_eval_examples
    #
    #         result = {'eval_accuracy': eval_accuracy}
    #
    #         output_eval_file = os.path.join(args.output_dir, "test_results.txt")
    #         with open(output_eval_file, "a", encoding='utf8') as writer:
    #             for key in sorted(result.keys()):
    #                 logger.info("  %s = %s", key, str(result[key]))
    #                 writer.write("%s = %s\n" % (key, str(result[key])))
    #             writer.write('*' * 80)
    #             writer.write('\n')
    #
    #         for key, value in result.items():
    #             tb_writer.add_scalar('test_{}'.format(key), value, global_step)

if __name__ == "__main__":
    main()
