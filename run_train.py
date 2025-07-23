from models import *
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
from utils import L2dis, L2ctr, L2ext, L2per, L2sym, L2ags, L2mix, L2dis_weighted, L2ctr_weighted, L2sym_weighted, L2per_weighted, L2ext_weighted, L2ags_weighted, L2mix_weighted
from types import SimpleNamespace
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    model = MPMC_net(args.dim, args.nhid, args.nlayers, args.nsamples, args.nbatch,
                     args.radius, args.loss_fn, args.weights, args.n_projections).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = 10000.
    patience = 0

    start_reduce = 100000
    reduce_point = 10

    Path('results/dim_' + str(args.dim) + '/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid)).mkdir(parents=True, exist_ok=True)
    Path('outputs/dim_' + str(args.dim) + '/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid)).mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(args.epochs), desc=f"Training: N={args.nsamples}, nhid={args.nhid}, loss={args.loss_fn}"):
        model.train()
        optimizer.zero_grad()
        loss, X = model()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            y = X.clone().detach()
            loss_func = globals()[args.loss_fn]

            #  take out of loop!!!
            if 'weighted' in args.loss_fn:
                batched_discrepancies = loss_func(y, args.weights)
            else:
                batched_discrepancies = loss_func(y)

            min_discrepancy = torch.min(batched_discrepancies).item()

            if min_discrepancy < best_loss:
                best_loss = min_discrepancy
                f = open('results/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid) + '/Lf'+str(args.loss_fn) + '.txt', 'a')
                f.write(str(best_loss) + '\n')
                f.close()

                PATH = 'outputs/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+ '/nhid_' +str(args.nhid)+ '/Lf'+str(args.loss_fn) + '.npy'
                y = y.detach().cpu().numpy()
                np.save(PATH,y)

            if (min_discrepancy > best_loss and (epoch + 1) >= args.start_reduce):
                patience += 1

            if (epoch + 1) >= args.start_reduce and patience == reduce_point:
                patience = 0
                args.lr /= 10.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr

            if (args.lr < 1e-6):
                f = open('results/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid) + '/Lf'+str(args.loss_fn) + '.txt', 'a')
                f.write('### epochs: '+str(epoch) + '\n')
                f.close()
                break


if __name__ == '__main__':
    #me: 'L2ext_weighted'
    #you/Lijia: 'L2per_weighted', 'L2sym_weighted'
    # loss_functions = ['L2sym_weighted', 'L2ags_weighted', 'L2mix_weighted']
    loss_functions = ['L2ext_weighted']

    for N in tqdm([16, 32, 64, 128, 256], desc='Sample Sizes'):
        for nh in tqdm([32], desc="Hidden Units", leave=False):
            for l in tqdm(loss_functions, desc="Loss Fn", leave=False):
                dim = 52

                # 52 dimensions for 52 weeks in a year
                gamma = torch.zeros(52, device=device)
                gamma[0:3] = 1.0

                args = {
                    'lr': 0.001,
                    'nlayers': 3,
                    'weight_decay': 1e-6,
                    'nhid': nh,
                    'nbatch': 1,
                    'epochs': 200000,
                    'start_reduce': 100000,
                    'radius': 0.35,
                    'nsamples': N,
                    'dim': dim,
                    'loss_fn': l,
                    'weights': gamma if 'weighted' in l else None,
                    'n_projections': 15
                }

                args = SimpleNamespace(**args)
                train(args)
                print(f"Finished training with N = {N}, nhid = {nh}, loss_fun = {l}")

"""from models import *
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
from utils import L2discrepancy, hickernell_all_emphasized, L2center, L2dis,  L2ext, L2per, L2sym, L2mix, L2avgs, L2ags_weighted, L2ctr_weighted, L2dis_weighted, L2ext_weighted, L2mix_weighted, L2per_weighted, L2sym_weighted
from types import SimpleNamespace
from tqdm import tqdm 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    print("intrain")
    model = MPMC_net(args.dim, args.nhid, args.nlayers, args.nsamples, args.nbatch,
                     args.radius, args.loss_fn, args.dim_emphasize, args.n_projections).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = 10000.
    patience = 0

    ## could be tuned for better performance
    start_reduce = 100000
    reduce_point = 10

    Path('results/dim_' + str(args.dim) + '/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid)).mkdir(parents=True, exist_ok=True)
    Path('outputs/dim_' + str(args.dim) + '/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid)).mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(args.epochs), desc = f"Training: N={args.nsamples}, nhid={args.nhid}, loss={args.loss_fn}"):
        model.train()
        optimizer.zero_grad()
        loss, X = model()
        loss.backward()
        optimizer.step()

        if epoch % 100 ==0:
            y = X.clone()
            if args.loss_fn == 'L2dis':
                batched_discrepancies = L2discrepancy(y.detach())
            elif args.loss_fn == 'L2cen':
                batched_discrepancies = L2center(y.detach())
            elif args.loss_fn == 'L2ext':
                batched_discrepancies = L2ext(y.detach())
            elif args.loss_fn == 'L2per':
                batched_discrepancies = L2per(y.detach())
            elif args.loss_fn == 'L2sym':
                batched_discrepancies = L2sym(y.detach())
            elif args.loss_fn == 'L2avgs':
                batched_discrepancies = L2avgs(y.detach())
            elif args.loss_fn == 'L2mix':
                batched_discrepancies = L2mix(y.detach())
            elif
            elif args.loss_fn == 'approx_hickernell':
                ## compute sum over all projections with emphasized dimensionality:
                batched_discrepancies = hickernell_all_emphasized(y.detach(),args.dim_emphasize)
            else:
                print('Loss function not implemented')
            min_discrepancy, mean_discrepancy = torch.min(batched_discrepancies).item(), torch.mean(batched_discrepancies).item()

            if min_discrepancy < best_loss:
                best_loss = min_discrepancy
                f = open('results/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid) + '/Lf'+str(args.loss_fn) + '.txt', 'a')
                f.write(str(best_loss) + '\n')
                f.close()

                ## save MPMC points:
                PATH = 'outputs/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+ '/nhid_' +str(args.nhid)+ '/Lf'+str(args.loss_fn) + '.npy'
                y = y.detach().cpu().numpy()
                np.save(PATH,y)

            if (min_discrepancy > best_loss and (epoch + 1) >= args.start_reduce):
                patience += 1

            if (epoch + 1) >= args.start_reduce and patience == reduce_point:
                patience = 0
                args.lr /= 10.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr

            if (args.lr < 1e-6):
                f = open('results/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid) + '/Lf'+str(args.loss_fn) + '.txt', 'a')
                f.write('### epochs: '+str(epoch) + '\n')
                f.close()
                break
            


if __name__ == '__main__':

    for N in tqdm([100], desc = 'Sample Sizes'):
        for nh in tqdm([32], desc = "Hidden Units", leave = False):
            for l in tqdm(['L2per', 'L2dis', 'L2avgs', 'L2ext', 'L2mix', 'L2per', 'L2sym'], desc = "Loss Fn", leave = False):
                args = {
                    'lr': 0.001,                  # learning rate
                    'nlayers': 3,                 # number of GNN layers
                    'weight_decay': 1e-6,         # weight decay (L2 regularization)
                    'nhid': nh,                  # number of hidden features in the GNN
                    'nbatch': 1,                  # number of point sets in a batch
                    'epochs': 200000,             # number of training epochs
                    'start_reduce': 100000,       # epoch to start reducing learning rate
                    'radius': 0.35,               # radius for GNN neighborhood
                    'nsamples': N,               # number of samples in each point set
                    'dim': 2,                     # dimensionality of the points
                    'loss_fn': l,           # loss function to use
                    'dim_emphasize': [1],         # emphasized dimensionalities for projections
                    'n_projections': 15           # number of projections (for approx_hickernell)
                }

                args = SimpleNamespace(**args)
                train(args)
                print(f"Finished training with N = {N}, nhid = {nh}, loss_fun = {l}")


    # parser = argparse.ArgumentParser(description='training parameters')
    
    # parser.add_argument('--lr', type=float, default=0.001,
    #                     help='number of samples')
    # parser.add_argument('--nlayers', type=int, default=3,
    #                     help='number of GNN nlayers')
    # parser.add_argument('--weight_decay', type=float, default=1e-6,
    #                     help='weight_decay')
    # parser.add_argument('--nhid', type=int, default=128,
    #                     help='number of hidden features of GNN')
    # parser.add_argument('--nbatch', type=int, default=1,
    #                     help='number of point sets in batch')
    # parser.add_argument('--epochs', type=int, default=200000,
    #                     help='number of epochs')
    # parser.add_argument('--start_reduce', type=int, default=100000,
    #                     help='when to start lr decay')
    # parser.add_argument('--radius', type=float, default=0.35,
    #                     help='radius for nearest neighbor GNN graph')
    # parser.add_argument('--nsamples', type=int, default=64,
    #                     help='number of samples')
    # parser.add_argument('--dim', type=int, default=2,
    #                     help='dimension of points')
    # parser.add_argument('--loss_fn', type=str, default='L2dis',
    #                     help='which loss function to use. Choices: ["L2dis","approx_hickernell"]')
    # parser.add_argument('--dim_emphasize', type=list, default=[1],
    #                     help='if loss_fn set to "approx_hickernell", specify which dimensionality to emphasize.'
    #                          'Note: It is not the coordinate of the points, but the dimension of the'
    #                          'projections, i.e., seeting dim_emphasize = [1,3] puts an emphasize'
    #                          'on 1-dimensional and 3-dimensional projections. Cannot emphasize all'
    #                          'dimensionalities.')
    # parser.add_argument('--n_projections', type=int, default=15,
    #                     help='number of projections for approx_hickernell')

    # args = parser.parse_args()
    # train(args)

"""