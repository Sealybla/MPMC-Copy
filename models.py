import math
import numpy as np
from itertools import product 
import torch
from torch import nn
from torch_cluster import radius_graph
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch_geometric.nn import MessagePassing, InstanceNorm
from utils import L2dis, L2ctr, L2ext, L2per, L2sym, L2ags, L2mix, L2dis_weighted, L2ctr_weighted, L2sym_weighted, L2per_weighted, L2ext_weighted, L2ags_weighted, L2mix_weighted


class MPNN_layer(MessagePassing):
    def __init__(self, ninp, nhid):
        super(MPNN_layer, self).__init__()
        self.ninp = ninp
        self.nhid = nhid

        self.message_net_1 = nn.Sequential(nn.Linear(2 * ninp, nhid),
                                           nn.ReLU()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(nhid, nhid),
                                           nn.ReLU()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(ninp + nhid, nhid),
                                          nn.ReLU()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(nhid, nhid),
                                          nn.ReLU()
                                          )
        self.norm = InstanceNorm(nhid)

    def forward(self, x, edge_index, batch):
        x = self.propagate(edge_index, x=x)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j):
        message = self.message_net_1(torch.cat((x_i, x_j), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        return update


class MPMC_net(nn.Module):
    def __init__(self, dim, nhid, nlayers, nsamples, nbatch, radius, loss_fn, weights, n_projections):
        super(MPMC_net, self).__init__()
        self.enc = nn.Linear(dim,nhid)
        self.convs = nn.ModuleList()
        for i in range(nlayers):
            self.convs.append(MPNN_layer(nhid,nhid))
        self.dec = nn.Linear(nhid,dim)
        self.nlayers = nlayers
        self.mse = torch.nn.MSELoss()
        self.nbatch = nbatch
        self.nsamples = nsamples
        self.dim = dim
        self.n_projections = n_projections
        self.enc = nn.Linear(dim,nhid)
        self.convs = nn.ModuleList()
        for i in range(nlayers):
            self.convs.append(MPNN_layer(nhid,nhid))
        self.dec = nn.Linear(nhid,dim)
        self.nlayers = nlayers
        self.mse = torch.nn.MSELoss()
        self.nbatch = nbatch
        self.nsamples = nsamples
        self.dim = dim
        self.n_projections = n_projections

        ## random shift
        self.x = torch.rand(nsamples * nbatch, dim).to(device)

        # store weights
        self.weights = weights

        batch = torch.arange(nbatch).unsqueeze(-1).to(device)
        batch = batch.repeat(1, nsamples).flatten()
        self.batch = batch
        self.edge_index = radius_graph(self.x, r=radius, loop=True, batch=batch).to(device)

        all_losses = {'L2dis', 'L2ctr', 'L2ext', 'L2per', 'L2sym', 'L2ags', 'L2mix', 'L2dis_weighted',
                      'L2ctr_weighted', 'L2ext_weighted', 'L2per_weighted', 'L2sym_weighted', 'L2ags_weighted','L2mix_weighted'}
        if loss_fn in all_losses:
            self.loss_fn = globals()[loss_fn]
        else:
            raise ValueError(f"Loss function DNE: {loss_fn}")

    def forward(self):
        X = self.x
        edge_index = self.edge_index

        X = self.enc(X)
        for i in range(self.nlayers):
            X = self.convs[i](X,edge_index,self.batch)
        X = torch.sigmoid(self.dec(X))
        X = X.view(self.nbatch, self.nsamples, self.dim)

        if self.weights is not None:
            loss = torch.mean(self.loss_fn(X, self.weights))
        else:
            loss = torch.mean(self.loss_fn(X))
        return loss, X



"""import torch
import math
from torch import nn
from torch_cluster import radius_graph
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch_geometric.nn import MessagePassing, InstanceNorm
import numpy as np
from itertools import product 

class MPNN_layer(MessagePassing):
    def __init__(self, ninp, nhid):
        super(MPNN_layer, self).__init__()
        self.ninp = ninp
        self.nhid = nhid

        self.message_net_1 = nn.Sequential(nn.Linear(2 * ninp, nhid),
                                           nn.ReLU()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(nhid, nhid),
                                           nn.ReLU()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(ninp + nhid, nhid),
                                          nn.ReLU()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(nhid, nhid),
                                          nn.ReLU()
                                          )
        self.norm = InstanceNorm(nhid)

    def forward(self, x, edge_index, batch):
        x = self.propagate(edge_index, x=x)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j):
        message = self.message_net_1(torch.cat((x_i, x_j), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        return update


class MPMC_net(nn.Module):
    def __init__(self, dim, nhid, nlayers, nsamples, nbatch, radius, loss_fn, dim_emphasize, n_projections):
        super(MPMC_net, self).__init__()
        self.enc = nn.Linear(dim,nhid)
        self.convs = nn.ModuleList()
        for i in range(nlayers):
            self.convs.append(MPNN_layer(nhid,nhid))
        self.dec = nn.Linear(nhid,dim)
        self.nlayers = nlayers
        self.mse = torch.nn.MSELoss()
        self.nbatch = nbatch
        self.nsamples = nsamples
        self.dim = dim
        self.n_projections = n_projections
        self.dim_emphasize = torch.tensor(dim_emphasize).long()

        ## random input points for transformation:
        self.x = torch.rand(nsamples * nbatch, dim).to(device)
        batch = torch.arange(nbatch).unsqueeze(-1).to(device)
        batch = batch.repeat(1, nsamples).flatten()
        self.batch = batch
        self.edge_index = radius_graph(self.x, r=radius, loop=True, batch=batch).to(device)

        if loss_fn == 'L2dis':
            self.loss_fn = self.L2discrepancy
        elif loss_fn == 'L2cen':
            self.loss_fn = self.L2center
        elif loss_fn == 'L2ext':
            self.loss_fn = self.L2ext
        elif loss_fn == 'L2per':
            self.loss_fn = self.L2per
        elif loss_fn == 'L2sym':
            self.loss_fn = self.L2sym
        elif loss_fn == 'L2avgs':
            self.loss_fn = self.L2avgs
        elif loss_fn == 'L2mix':
            self.loss_fn = self.L2mix
        elif loss_fn == 'approx_hickernell':
            if dim_emphasize != None:
                assert torch.max(self.dim_emphasize) <= dim
                self.loss_fn = self.approx_hickernell
        else:
            raise ValueError(f"Loss function DNE: {loss_fn}")

    def approx_hickernell(self, X):
        X = X.view(self.nbatch, self.nsamples, self.dim)
        disc_projections = torch.zeros(self.nbatch).to(device)

        for i in range(self.n_projections):
            ## sample among non-emphasized dimensionality
            mask = torch.ones(self.dim, dtype=bool)
            mask[self.dim_emphasize - 1] = False
            remaining_dims = torch.arange(1, self.dim + 1)[mask]
            projection_dim = remaining_dims[torch.randint(low=0, high=remaining_dims.size(0), size=(1,))].item()
            projection_indices = torch.randperm(self.dim)[:projection_dim]
            disc_projections += self.L2discrepancy(X[:, :, projection_indices])
            ## sample among emphasized dimensionality
            remaining_dims = torch.arange(1, self.dim + 1)[self.dim_emphasize - 1]
            projection_dim = remaining_dims[torch.randint(low=0, high=remaining_dims.size(0), size=(1,))].item()
            projection_indices = torch.randperm(self.dim)[:projection_dim]
            disc_projections += self.L2discrepancy(X[:, :, projection_indices])

        return disc_projections

    # def returnx(self):
    #     return self.x
    
    # def bark(self):
    #     print("woof")


    def L2discrepancy(self, x):
        N = x.size(1) 
        dim = x.size(2)
        prod1 = 1. - x ** 2.
        prod1 = torch.prod(prod1, dim=2) #multiplying across second dimenstion of x (dim)
        sum1 = torch.sum(prod1, dim=1) #summing across second dimension of x (number of points in each batch)
        pairwise_max = torch.maximum(x[:, :, None, :], x[:, None, :, :])
        product = torch.prod(1 - pairwise_max, dim=3)
        sum2 = torch.sum(product, dim=(1, 2))
        one_dive_N = 1. / N
        out = torch.sqrt(
            math.pow(3., -dim) 
            - one_dive_N * math.pow(2., 1. - dim) * sum1 
            + 1. / math.pow(N, 2.) * sum2)
        return out

    def L2center(self, x):
        n = x.size(1)
        dim = x.size(2)

            # Term 1: (1/12)^d
        term1 = (1.0 / 12.0) ** dim

            # Term 2: -2/n * sum_i ∏_j (0.5|x_ij - 0.5| - 0.5|x_ij - 0.5|^2)
        sum1 = torch.abs(x - 0.5)
        prod1 = 0.5 * sum1 - 0.5 * (sum1 ** 2)
        prod2 = torch.prod(prod1, dim=2)
        sum2 = torch.sum(prod2, dim=1)
        term2 = -(2.0 / n) * sum2

            # Term 3: 1/(n^2) * sum_{i,j} ∏_k (0.5|x_ik - 0.5| + 0.5|x_jk - 0.5| - 0.5|x_ik - x_jk|)
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        prod3 = (
            0.5 * torch.abs(x_i - 0.5)
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
        
    
    def L2ext(self, x):
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
    
    def L2per(self, x):
        N = x.size(1)
        dim = x.size(2)

        prod2 = 0.5 - torch.abs(x[: ,: ,None ,: ] - x[: ,None ,: ,: ]) + (x[: ,: ,None ,: ] - x[: ,None ,: ,: ])**2
        product = torch.prod(prod2, dim = 3)
        sum2 = torch.sum(product, dim = (1,2))

        out = torch.sqrt(- math.pow(3., -dim) + math.pow(N, - 2.) * sum2)
        return out
    
    def L2sym(self, x): 
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

        
    def L2avgs (self, x):
        N = x.size(1)
        dim = x.size(2)
    
        #Term 1: (1/3) ^ d
        term1 = (1 / 3) ** dim
    
        # Term 2: -2/n * sum_i ∏_j (1 + 2x_ij + 2x_ij ^ 2) / 4
        sum1 = 2. * x - 2. * x**2
        sum2 = 1.0 + sum1 
        prod1 = sum2 / 4.
        prod2 = torch.prod(prod1, dim=2)
        sum2 = torch.sum(prod2, dim=1)
        term2 = -(2.0 / N) * sum2
    
        # Term 3: 1/(n^2) * sum_i sum_j ∏_k (1 -|x_ik - xj-jk|/2)
        x_i = x.unsqueeze(2) 
        x_j = x.unsqueeze(1) 
        sum3 = torch.abs(x_i - x_j)
        prod3 = torch.prod( (1.0 - sum3) / 2.0, dim=3)
        term3_sum = torch.sum(prod3, dim=(1, 2))
        term3 = (1.0 / (N * N)) * term3_sum
       
        out = torch.sqrt(term1+ term2 + term3)
        return out
    
    def L2mix(self, x):
     N = x.size(1)
     dim = x.size(2)
     prod1 = 2/3 - 1/4 * (torch.abs(x - 1/2)) - 1/4 * ((x - 1/2)**2)
     prod1 = torch.prod(prod1, dim = 2)
     sum1 = torch.sum(prod1, dim = 1)

     prod2 = 7/8 - 1/4 * torch.abs(x[: ,: ,None ,: ] - 1/2) - 1/4 * torch.abs(x[:, None, :, :] - 1/2) - 3/4*torch.abs(x[: ,: ,None ,: ] - x[:, None, :, :]) + 1/2 * torch.pow(x[: ,: ,None ,: ] - x[:, None, :, :], 2)
     product = torch.prod(prod2, dim = 3)
     sum2 = torch.sum(product, dim = (1,2))

     out = torch.sqrt(math.pow(7./12., dim) - (2. / N) * sum1 + math.pow(N, -2.) * sum2)
     return out
    



    def L2ags_weighted (self, x, gamma):
        N = x.size(1)
        dim = x.size(2)

        term1 = torch.prod(1.0 + gamma**2 / 3.0)

        g_term2 = gamma.view(1, 1, dim)
        sum1 = g_term2 * (2. * x - 2. * x**2)
        sum2 = 1.0 + sum1
        prod1 = sum2 / 4. 
        prod2 = torch.prod(prod1, dim=2)
        sum_prod2 = torch.sum(prod2, dim=1)
        term2 = -(2. / N) * sum_prod2

        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        g_term3 = gamma.view(1, 1, 1, dim)   # need to reshape?
        sum3 = g_term3 * torch.abs(x_i - x_j)
        prod3 = torch.prod((1. - sum3) / 2., dim=3)
        term3_sum = torch.sum(prod3, dim=(1, 2))
        term3 = (1. / (N * N)) * term3_sum

        out_sq = term1 + term2 + term3
        out = torch.sqrt(torch.relu(out_sq))
        
        return out

    
    def forward(self):
        X = self.x
        edge_index = self.edge_index

        X = self.enc(X)
        for i in range(self.nlayers):
            X = self.convs[i](X,edge_index,self.batch)
        X = torch.sigmoid(self.dec(X))  ## clamping with sigmoid needed so that warnock's formula is well-defined
        X = X.view(self.nbatch, self.nsamples, self.dim)
        loss = torch.mean(self.loss_fn(X))
        return loss, X
    
"""