import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64 # 64 blocks per iteration
block_size = 256 # 256 tokens per block
max_iters = 5000 # 5000 iterations through the network
eval_interval = 500 # every 500 iterations, test on val data
learning_rate = 3e-4 # how much to adjust weights and biases 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # how many batches of data are used to evaluate the loss

