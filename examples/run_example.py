# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# An example for loading Roberta with adapters

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
from pytorch_transformers.my_modeling_roberta import RobertaModelwithAdapter
from pytorch_transformers import (RobertaConfig,RobertaTokenizer,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default='./proc/roberta_adapter', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--freeze_bert", default=False, type=bool,
                        help="freeze the parameters of original model.")
    parser.add_argument("--freeze_adapter", default=True, type=bool,
                        help="freeze the parameters of adapter.")

    parser.add_argument('--fusion_mode', type=str, default='concat',
                        help='the fusion mode for bert feature and adapter feature |add|concat')

    parser.add_argument("--adapter_transformer_layers", default=2, type=int,
                        help="The transformer layers of adapter.")
    parser.add_argument("--adapter_size", default=768, type=int,
                        help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,11,23", type=str,
                        help="The layer where add an adapter")
    parser.add_argument("--adapter_skip_layers", default=0, type=int,
                        help="The skip_layers of adapter according to bert layers")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--meta_fac_adaptermodel', default="../pretrained_models/fac-adapter/pytorch_model.bin" ,type=str, help='the pretrained factual adapter model')
    parser.add_argument('--meta_lin_adaptermodel', default="../pretrained_models/lin-adapter/pytorch_model.bin", type=str, help='the pretrained linguistic adapter model')
    args = parser.parse_args()

    args.adapter_list = args.adapter_list.split(',')
    args.adapter_list = [int(i) for i in args.adapter_list]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    ## test loading, inference, and save model, load model.
    # Load pretrained model/tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaModelwithAdapter(args)

    model.to(args.device)

    # Encode text
    input_ids = torch.tensor([tokenizer.encode("Here is some text to encode",
                                               add_special_tokens=True)]).to(args.device)  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are tuples

    # The last_hidden_state can be used as input for downstream tasks
    print('last_hidden_states:',last_hidden_states)

    # Savning model
    print('Saving model...')
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)

    # Loading saved model
    print('Loading model...')
    if hasattr(model, 'module'):
        model.module.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model.bin')))
    else:  # Take care of distributed/parallel training
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model.bin')))

if __name__ == "__main__":
    main()


