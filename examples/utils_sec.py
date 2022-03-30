""" BERT regression fine-tuning: utilities to work with SEC reports"""

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
import json
import numpy as np

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        # self.start_id = start_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            return json.load(f)

    @classmethod
    def _read_semeval_txt(clas, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            examples = []
            example = []
            for line in f:
                if line.strip() == '':
                    examples.append(example)
                    example = []
                else:
                    example.append(line.strip())
            return examples
        

class SECProcessor(DataProcessor):
    """Processor for our SEC filings data"""
    def get_train_examples(self, data_dir, dataset_type=None):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir, dataset_type):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type)

    def get_labels(self):
        """See base class."""
        return 0

    def _create_examples(self, list_of_dicts, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        # label_list = ['entity', 'location', 'time', 'organization', 'object', 'event', 'place', 'person', 'group']
        for (i, curr_dict_input) in enumerate(list_of_dicts):
            if 'risk_paragraphs' not in curr_dict_input.keys():
                continue
            
            for k, mda_paragraph in curr_dict_input['risk_paragraphs'].items():
                
            
                guid = i
                text_a = mda_paragraph
                # text_b = (line['start'], line['end'])
                # label = [0 for item in range(len(label_list))]
                # for item in line['labels']:
                #     label[label_list.index(item)] = 1
                label = curr_dict_input['percentage_change']

                examples.append(
                    InputExample(guid=guid, text_a=text_a, label=label))
        return examples
    

# Modified from entity typing
def convert_examples_to_features_sec(examples, max_seq_length,
                                               tokenizer, output_mode,
                                               cls_token='[CLS]',
                                               sep_token='[SEP]',
                                               pad_on_left=False,
                                               pad_token=0,
                                               pad_token_segment_id=0,
                                               sequence_a_segment_id=1,
                                               mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    # check if pad_token_segment_id should be 0
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # start, end = example.text_b[0], example.text_b[1]
        sentence = example.text_a
        tokens_sentence = tokenizer.tokenize(sentence)
        # truncate if needed
        tokens_sentence = [cls_token] + tokens_sentence[:max_seq_length-2] + [sep_token]
        # tokens_sentence = [cls_token] + tokens_sentence + [sep_token]
        # tokens_0_start = tokenizer.tokenize(sentence[:start])
        # tokens_start_end = tokenizer.tokenize(sentence[start:end])
        # tokens_end_last = tokenizer.tokenize(sentence[end:])
        # tokens = [cls_token] + tokens_0_start + tokenizer.tokenize('@') + tokens_start_end + tokenizer.tokenize(
        #     '@') + tokens_end_last + [sep_token]
        # start = 1 + len(tokens_0_start)
        # end = 1 + len(tokens_0_start) + 1 + len(tokens_start_end)

        # segment_ids = [sequence_a_segment_id] * len(tokens)
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        segment_ids = [sequence_a_segment_id] * len(tokens_sentence)
        input_ids = tokenizer.convert_tokens_to_ids(tokens_sentence)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = example.label
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens_sentence]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))
        # start_id = np.zeros(max_seq_length)
        # start_id[start] = 1
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features
    
    
processors = {
    "sec_regressor": SECProcessor
}

output_modes = {
    "sec_regressor": "regression"
}

SEC_TASKS_NUM_LABELS = {
    "sec_regressor": 1
}