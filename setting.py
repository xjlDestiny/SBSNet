

import random

seeds = [23]

NUM_CLUSTERS = 10
NUM_PER_CLUSTER = 64
TEST_BATCH_SIZE = 128
type_target_spectrum = 2
epochs = 200


# 模型参数
MODEL_NAME = 'MSSAE'
mapped_len = 128
d_input = 1
d_model = 32
layers = 2
# TE
ffn_hidden = 128
n_head = 2
n_layers = 1

step_decay = 10
lr = 1e-4
weight_decay = 0
gamma = 0.75
encode_f = 0.8
