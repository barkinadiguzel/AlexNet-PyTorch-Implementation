import torch.nn as nn

def lrn_layer():
    # size=n, alpha, beta, k
    return nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
