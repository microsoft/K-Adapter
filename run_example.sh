# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# An example for testing RoBERTa with adapters: define, inference, save and load.

# ./pretrained_models/lin-adapter/pytorch_model.bin"
python examples/run_example.py \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,23" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" \
    --meta_lin_adaptermodel=""
