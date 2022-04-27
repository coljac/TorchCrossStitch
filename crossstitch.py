import sys
import os
import numpy as np
import torch
import torch.nn as nn


class CrossStitchUnit(nn.Module):
    """ Cross stitch unit per Misra 2016. 
    
        Parameters:
            n (int): The number of sub-networks joined by this cross-stitch
            unit.

        Returns:
            Instance of cross-stitch Module with n^2 weights.
    """ 
    def __init__(self, n=2):
        super().__init__()
        weights = torch.Tensor(np.identity(n)*(.9-.1/(n-1)) + .1/(n-1))
        self.weights = nn.Parameter(data=weights) 
        
    def forward(self, x):
        res = self.weights.t().unsqueeze(0).repeat(x.shape[0], 1, 1)
        res = x.bmm(res)
        return res


class FlexibleCrossStitch(nn.Module):
    """
        Creates a feed-forward ANN from supplied components, with cross-stitch units 
        in between. The FlexibleCrossStitch creates n copies of the supplied sub-modules,
        joined by cross-stitch units. Thus, at the input layer the input is passed to 
        each of the n Module instances at the first position; the outputs are weighted 
        by the cross-stitch unit, then passed to the next layer of child modules. The 
        final outputs are returned as a single vector.

        Parameters:
          nets (list): A list of functions that instantiate instance of nn.Module. Each
                       child module is instantiated split times. 
          split (int): The number of networks to be executed in parallel and joined by 
                       cross-stitch units.

          Returns:
            A feed forward neural network with cross-stitch units joining the cloned 
            submodules.
    """
    def __init__(self, nets, split=2):
        super().__init__()
        self.split = split
        self.sub_count = len(nets)
        for i, net in enumerate(nets):
            for j in range(split):
                setattr(self, f"net_{i+1}_{j+1}", net())    
            if i < len(nets)-1:
                setattr(self, f"cs{i+1}", CrossStitchUnit(n=split))    
        
    def forward(self, x):
        if x.dim() < 2:
            x = torch.unsqueeze(x, 0) # Batch size of 1
        x = torch.unsqueeze(x, 2).repeat(1, 1, self.split)
        for i in range(self.sub_count):
            results = torch.zeros((x.shape[0], getattr(self, f"net_{i+1}_{1}").output_size, self.split))
            if x.is_cuda:
                results = results.cuda()
            for j in range(self.split):
                results[:,:, j] = getattr(self, f"net_{i+1}_{j+1}")(x[:, :, j])
            if i < self.sub_count-1:
                results = getattr(self, f"cs{i+1}")(results)
            x = results
        results = results[:, 0, :]
        return results
