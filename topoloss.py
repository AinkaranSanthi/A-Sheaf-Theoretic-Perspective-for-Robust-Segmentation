from multiprocessing import Pool
import cripser as crip
import tcripser as trip
import numpy as np
import torch
import torch.nn.functional as F
import cripser
import copy
from torch.optim import SGD, Adam


#This python module is adapated from code from the paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9872052
def crip_wrapper(X, D):
    return crip.computePH(X, maxdim=D)

def trip_wrapper(X, D):
    return trip.computePH(X, maxdim=D)

def get_differentiable_barcode(tensor, barcode):
        '''Makes the barcode returned by CubicalRipser differentiable using PyTorch.
        Note that the critical points of the CubicalRipser filtration reveal changes in sub-level set topology.
    
        Arguments:
        REQUIRED
        tensor  - PyTorch tensor w.r.t. which the barcode must be differentiable
        barcode - Barcode returned by using CubicalRipser to compute the PH of tensor.numpy() 
        '''
        # Identify connected component of ininite persistence (the essential feature)
        inf = barcode[barcode[:, 2] == np.finfo(barcode.dtype).max]
        fin = barcode[barcode[:, 2] < np.finfo(barcode.dtype).max]
    
        # Get birth of infinite feature
        inf_birth = tensor[tuple(inf[:, 3:3+tensor.ndim].astype(np.int64).T)]
    
        # Calculate lifetimes of finite features
        births = tensor[tuple(fin[:, 3:3+tensor.ndim].astype(np.int64).T)]
        deaths = tensor[tuple(fin[:, 6:6+tensor.ndim].astype(np.int64).T)]
        delta_p = (deaths - births)
    
        # Split finite features by dimension
        delta_p = [delta_p[fin[:, 0] == d] for d in range(tensor.ndim)]
    
        # Sort finite features by persistence
        delta_p = [torch.sort(d, descending=True)[0] for d in delta_p]
    
    
        return inf_birth, delta_p
def topo_loss(
        outputs, prior,
        mse_lambda=1000,
        construction='0', thresh=None, parallel=True):
        # Get image properties
        spatial_xyz = list(outputs[0].shape)
        outputs_roi = outputs
    
    
        # Get working device
        device = 'cuda'

        combos = [1 - outputs_roi[i] for i in range(len(outputs_roi))]
        max_dims = [len(b) for b in prior.values()]
        prior = {torch.tensor(c): torch.tensor(b) for c, b in prior.items()}

        PH = {'0': crip_wrapper, 'N': trip_wrapper}

        # Get barcodes using cripser in parallel without autograd            
        combos_arr = [combos[i].detach().cpu().numpy().astype(np.float64) for i in range(len(combos))]
        if parallel:
            with torch.no_grad():
                with Pool(len(prior)) as p:
                    bcodes_arr = p.starmap(PH[construction], zip(combos_arr, max_dims))
        else:
            with torch.no_grad():
                bcodes_arr = [PH[construction](combo, max_dim) for combo, max_dim in zip(combos_arr, max_dims)]

        # Get differentiable barcodes using autograd
        max_features = max([bcode_arr.shape[0] for bcode_arr in bcodes_arr])
        bcodes = torch.zeros([len(prior), max(max_dims)+1, max_features], requires_grad=False, device=device)
        for c, (combo, bcode) in enumerate(zip(combos, bcodes_arr)):
            
            _, fin = get_differentiable_barcode(combo, bcode)
            for dim in range(len(spatial_xyz)):
                bcodes[c, dim, :len(fin[dim])] = fin[dim]

        # Select features for the construction of the topological loss
        stacked_prior = torch.stack(list(prior.values()))
        stacked_prior.T[0] -= 1 # Since fundamental 0D component has infinite persistence
        matching = torch.zeros_like(bcodes).detach().bool()
        for c, combo in enumerate(stacked_prior):
            for dim in range(len(combo)):
                matching[c, dim, slice(None, stacked_prior[c, dim])] = True

        # Find total persistence of features which match (A) / violate (Z) the prior
        A = (1 - bcodes[matching]).sum()
        Z = bcodes[~matching].sum()

        loss = A + Z                     


        return loss