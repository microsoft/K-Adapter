import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn

from pytorch_transformers import BertModel, BasicTokenizer

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer as BertTokenizerLocal
from transformers import BertTokenizer as BertTokenizerHugging

from utils_sec import output_modes, processors, convert_examples_to_features_sec

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, task, tokenizer, dataset_type, evaluate=False):
    # Modify for our dataset
    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            dataset_type,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir, dataset_type)
            if evaluate
            else processor.get_train_examples(args.data_dir, dataset_type)
        )
        features = convert_examples_to_features_sec(
            examples,
            args.max_seq_length,
            tokenizer,
            output_mode,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
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
    # all_start_ids = torch.tensor([f.start_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )
    return dataset


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class PretrainedModel(nn.Module):
    def __init__(self, args):
        super(PretrainedModel, self).__init__()
        self.model = BertModel.from_pretrained(
            args.finbert_path, output_hidden_states=True
        )
        self.config = self.model.config
        self.config.freeze_adapter = args.freeze_adapter
        if args.freeze_bert:
            for p in self.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        start_id=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        return outputs  # (loss), logits, (hidden_states), (attentions)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--finbert_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: "
        + ", ".join(processors.keys()),
    )
    parser.add_argument("--comment", default="", type=str, help="The comment")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--freeze_bert",
        default=True,
        type=bool,
        help="freeze the parameters of pretrained model.",
    )
    parser.add_argument(
        "--freeze_adapter",
        default=False,
        type=bool,
        help="freeze the parameters of adapter.",
    )

    parser.add_argument("--test_mode", default=0, type=int, help="test freeze adapter")

    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="concat",
        help="the fusion mode for bert feautre and adapter feature |add|concat",
    )
    parser.add_argument(
        "--adapter_transformer_layers",
        default=2,
        type=int,
        help="The transformer layers of adapter.",
    )
    parser.add_argument(
        "--adapter_size", default=768, type=int, help="The hidden size of adapter."
    )
    parser.add_argument(
        "--adapter_list",
        default="0,11,22",
        type=str,
        help="The layer where add an adapter",
    )
    parser.add_argument(
        "--adapter_skip_layers",
        default=3,
        type=int,
        help="The skip_layers of adapter according to bert layers",
    )

    parser.add_argument(
        "--meta_fac_adaptermodel",
        default="",
        type=str,
        help="the pretrained factual adapter model",
    )
    parser.add_argument(
        "--meta_et_adaptermodel",
        default="",
        type=str,
        help="the pretrained entity typing adapter model",
    )
    parser.add_argument(
        "--meta_lin_adaptermodel",
        default="",
        type=str,
        help="the pretrained linguistic adapter model",
    )

    ## Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--meta_bertmodel", default="", type=str, help="the pretrained bert model"
    )
    parser.add_argument(
        "--save_model_iteration", type=int, help="when to save the model.."
    )

    args = parser.parse_args()

    args.adapter_list = args.adapter_list.split(",")
    args.adapter_list = [int(i) for i in args.adapter_list]

    name_prefix = (
        "batch-"
        + str(args.per_gpu_train_batch_size)
        + "_"
        + "lr-"
        + str(args.learning_rate)
        + "_"
        + "warmup-"
        + str(args.warmup_steps)
        + "_"
        + "epoch-"
        + str(args.num_train_epochs)
        + "_"
        + str(args.comment)
    )
    args.my_model_name = args.task_name + "_" + name_prefix
    args.output_dir = os.path.join(args.output_dir, args.my_model_name)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    # args.task_name = args.task_name.lower()
    # if args.task_name not in processors:
    #     raise ValueError("Task not found: %s" % (args.task_name))
    # processor = processors[args.task_name]()
    # args.output_mode = output_modes[args.task_name]
    # label_list = processor.get_labels()
    # num_labels = 1

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # args.model_type = args.model_type.lower()
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    # tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    # model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    tokenizer_hugging = BertTokenizerHugging.from_pretrained(
        "yiyanghkust/finbert-tone", model_max_length=args.max_seq_length
    )
    tokenizer_local = BertTokenizerLocal.from_pretrained("bert")
    pretrained_model = PretrainedModel(args)
    # if args.meta_fac_adaptermodel:
    #     fac_adapter = AdapterModel(args, pretrained_model.config)
    #     fac_adapter = load_pretrained_adapter(fac_adapter,args.meta_fac_adaptermodel)
    # else:
    #     fac_adapter = None
    # if args.meta_et_adaptermodel:
    #     et_adapter = AdapterModel(args, pretrained_model.config)
    #     et_adapter = load_pretrained_adapter(et_adapter,args.meta_et_adaptermodel)
    # else:
    #     et_adapter = None
    # if args.meta_lin_adaptermodel:
    #     lin_adapter = AdapterModel(args, pretrained_model.config)
    #     lin_adapter = load_pretrained_adapter(lin_adapter,args.meta_lin_adaptermodel)
    # else:
    #     lin_adapter = None
    # adapter_model = AdapterModel(pretrained_model.config,num_labels,args.adapter_size,args.adapter_interval,args.adapter_skip_layers)

    # regressor_model = RegressorModel(args, pretrained_model.config, fac_adapter=fac_adapter, et_adapter=et_adapter, lin_adapter=lin_adapter)

    # if args.meta_bertmodel:
    #     model_dict = pretrained_model.state_dict()
    #     bert_meta_dict = torch.load(args.meta_bertmodel, map_location=lambda storage, loc: storage)
    #     for item in ['out_proj.weight', 'out_proj.bias', 'dense.weight', 'dense.bias', 'lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias',
    #                  'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight']:
    #         if item in bert_meta_dict:
    #             bert_meta_dict.pop(item)

    #     changed_bert_meta = {}
    #     for key in bert_meta_dict.keys():
    #         changed_bert_meta[key.replace('model.','roberta.')] = bert_meta_dict[key]
    #     # print(changed_bert_meta.keys())
    #     changed_bert_meta = {k: v for k, v in changed_bert_meta.items() if k in model_dict.keys()}
    #     # print(changed_bert_meta.keys())
    #     model_dict.update(changed_bert_meta)
    #     pretrained_model.load_state_dict(model_dict)
    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # pretrained_model.to(args.device)
    # regressor_model.to(args.device)
    # # model.to(args.device)
    # model = (pretrained_model, regressor_model)

    # logger.info("Training/evaluation parameters %s", args)

    # # Training
    # if args.do_train:
    #     train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, 'train', evaluate=False)
    #     global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    #     logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)

    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = regressor_model.module if hasattr(regressor_model, 'module') else regressor_model  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)

    # # Evaluation
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         result = evaluate(args, model, tokenizer, prefix=global_step)
    #         logger.info('micro f1:{}'.format(result))
    #         result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #         results.update(result)
    # save_result = str(results)
    # save_results.append(save_result)
    #
    # result_file = open(os.path.join(args.output_dir, args.my_model_name + '_result.txt'), 'w')
    # for line in save_results:
    #     result_file.write(str(line) + '\n')
    # result_file.close()

    # return results


if __name__ == "__main__":
    main()
