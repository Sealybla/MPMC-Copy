import torch
import math
from itertools import combinations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def L2discrepancy(x):
    N = x.size(1)
    dim = x.size(2)
    prod1 = 1. - x ** 2.
    prod1 = torch.prod(prod1, dim=2)
    sum1 = torch.sum(prod1, dim=1)
    pairwise_max = torch.maximum(x[:, :, None, :], x[:, None, :, :])
    product = torch.prod(1 - pairwise_max, dim=3)
    sum2 = torch.sum(product, dim=(1, 2))
    one_dive_N = 1. / N
    out = torch.sqrt(math.pow(3., -dim) - one_dive_N * math.pow(2., 1. - dim) * sum1 + 1. / math.pow(N, 2.) * sum2)
    return out

def hickernell_all_emphasized(x,dim_emphasize):
    nbatch, nsamples, dim = x.size(0), x.size(1), x.size(2)
    mean_disc_projections = torch.zeros(nbatch).to(device)
    for d in dim_emphasize:
        subsets_of_d = list(combinations(range(dim), d))
        for i in range(len(subsets_of_d)):
            set_inds = subsets_of_d[i]
            mean_disc_projections += L2discrepancy(x[:, :, set_inds])

    return mean_disc_projections


def L2center(x):
        n = x.size(1)
        dim = x.size(2)

        # Term 1: (13/12)^d
        term1 = (13.0 / 12.0) ** dim

        # Term 2: -2/n * sum_i ∏_j (1 + 0.5|x_ij - 0.5| - 0.5|x_ij - 0.5|^2)
        sum1 = torch.abs(x - 0.5)
        prod1 = 1.0 + 0.5 * sum1 - 0.5 * (sum1 ** 2)
        prod2 = torch.prod(prod1, dim=2)
        sum2 = torch.sum(prod2, dim=1)
        term2 = -(2.0 / n) * sum2

        # Term 3: 1/(n^2) * sum_{i,j} ∏_k (1 + 0.5|x_ik - 0.5| + 0.5|x_jk - 0.5| - 0.5|x_ik - x_jk|)
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        prod3 = (
            1.0
            + 0.5 * torch.abs(x_i - 0.5)
            + 0.5 * torch.abs(x_j - 0.5)
            - 0.5 * torch.abs(x_i - x_j)
        )
        prod4 = torch.prod(prod3, dim=3)
        sum3 = torch.sum(prod4, dim=(1, 2))
        term3 = (1.0 / (n * n)) * sum3

        # Combine terms for squared discrepancy
        squared_discrepancy = term1 + term2 + term3

        # Return L2 discrepancy (square root of squared discrepancy)
        out = torch.sqrt(squared_discrepancy)
        return out
    
def L2ext(x):
    N = x.size(1)
    dim = x.size(2)

    prod1 = 0.5*(x - x**2.)
    prod1 = torch.prod(prod1, dim = 2)
    sum1 = torch.sum(prod1, dim = 1)

    prod2 = torch.min(x[: ,: ,None ,: ], x[: ,None ,: ,: ]) - x[: ,: ,None ,: ] * x[: ,None ,: ,: ]
    product = torch.prod(prod2, dim = 3)
    sum2 = torch.sum(product, dim = (1,2))

    out = torch.sqrt(math.pow(12., -dim) - (2. / N) * sum1 + math.pow(N, - 2.) * sum2)
    return out
    
def L2per(x):
    N = x.size(1)
    dim = x.size(2)

    prod2 = 0.5 - torch.abs(x[: ,: ,None ,: ] - x[: ,None ,: ,: ]) + (x[: ,: ,None ,: ] - x[: ,None ,: ,: ])**2
    product = torch.prod(prod2, dim = 3)
    sum2 = torch.sum(product, dim = (1,2))

    out = torch.sqrt(- math.pow(3., -dim) + math.pow(N, - 2.) * sum2)
    return out
    
def L2sym(x): 
    N = x.size(1)
    dim = x.size(2)

    prod1 = 0.5*(x - x**2.)
    prod1 = torch.prod(prod1, dim = 2)
    sum1 = torch.sum(prod1, dim = 1)

    prod2 = 0.25 * (1 - 2 * torch.abs(x[: ,: ,None ,: ] - x[: ,None ,: ,: ]))
    product = torch.prod(prod2, dim = 3)
    sum2 = torch.sum(product, dim = (1,2))

    out = torch.sqrt(math.pow(12., -dim) - (2. / N) * sum1 + math.pow(N, - 2.) * sum2)
    return out
