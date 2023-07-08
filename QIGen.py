from copy import deepcopy
import torch
from tqdm import tqdm
import gc

import cQIGen as qinfer
import math
import numpy as np
from gekko import GEKKO
from utils import mem_model

params = {}

def compute_reductions(x, gs=-1, cpp=True):
    if cpp:
        if len(x.shape) != 1:
            rows, cols = x.shape
        else:
            rows = 1
            cols = x.shape[0]
        if gs == -1:
            out = torch.zeros(rows).float().contiguous()
            mygs = cols
        else:
            out = torch.zeros(rows, cols // gs).float().contiguous()
            mygs = gs
        
        qinfer.compute_reduction_cpp(x, out, rows, cols, mygs)
        return out
    if gs == -1: 
        if len(x.shape) != 1:
            return torch.sum(x,1)
        else:
            return torch.sum(x)
    else:
        if len(x.shape) != 1:
            rows, cols = x.shape
            out = torch.zeros(rows, cols // gs).float().contiguous()
            for i in range(cols // gs):
                out[:,i] = torch.sum(x[:,i*gs:(i+1)*gs],1)
            return out
        else:
            cols = x.shape[0]
            out = torch.zeros(cols // gs).float().contiguous()
            for i in range(cols // gs):
                out[i] = torch.sum(x[i*gs:(i+1)*gs])
            return out

def process_zeros_scales(zeros, scales, bits, M):
    if zeros.dtype != torch.float32:
        new_zeros = torch.zeros_like(scales).float().contiguous()
        if bits == 4:
            qinfer.unpack_zeros4(zeros, new_zeros, new_zeros.shape[0], new_zeros.shape[1])
        elif bits == 2:
            qinfer.unpack_zeros2(zeros, new_zeros, new_zeros.shape[0], new_zeros.shape[1])
        elif bits == 3:
            print("Unpacking zeros for 3 bits")
        new_scales = scales.contiguous()
    else:
        if scales.shape[1] != M:
            new_scales = scales.transpose(0,1).contiguous()
        else:
            new_scales = scales.contiguous()
        if zeros.shape[1] != M:
            new_zeros = zeros.transpose(0,1).contiguous()
        else:
            new_zeros = zeros.contiguous()

    return new_zeros, new_scales
    
class qLinear(torch.nn.Module):
    def __str__(self):
        return self.name
    
    def __init__(self, mode, p, l1, name="", other=None, N=0, M=0, qweights=None, zeros=None, scales=None, bias=None, bits=4, hint=1, verbose=False, gs=-1):
        super().__init__()
        self.bits = bits
        pack = 32 // bits

        if mode == 'linear':
            self.N, self.M = other.in_features, other.out_features
        else:
            self.N, self.M = N, M

        n = hint
        m = self.N
        t = self.M
        

        #registers for now are fixed
        if bits == 3:
            packed = 32
            unroll = 3
            nu = 1 #args.n
            mu = 32
            tu = 32
        else:
            packed = 32 // bits
            unroll = 2
            nu = 1 #args.n
            mu = 16
            tu = 32
        
        nb = n # it's always small for transformers

        global params
        if (m,t) in params:
            mb = params[(m,t)][0]
            tb = params[(m,t)][1]
        else:
            if verbose:
                print("Computing memory model for {}x{}x{} with {} bits".format(n,m,t,bits))
            mb, tb = mem_model(n, m, t, mu, tu, bits, l1, p, gs, verbose=False)
            params[(m,t)] = (mb,tb)

        split = np.ones(p)
        split = split * tb
        while np.sum(split) < t:
            split = split + tb

        idx = p - 1
        while np.sum(split) > t:
            split[idx] = split[idx] - tb
            idx = idx - 1

        assert(np.sum(split) == t)

        split = split.astype(int)
        self.tt = int(split[0])

        if split[0] == split[-1]:
            self.cutoff = int(p+1)
        else:
            self.cutoff = int(idx + 1)

        self.mb = mb #// packed
        self.tb = tb

        self.gs = gs

        if verbose:
            print("Chose parameters {}x{}x{} with {} bits and tt {}".format(nb,mb,tb,bits,self.tt))


        self.name=name
        if bias is None:
            self.bias = torch.zeros(self.M)
        else:
            self.bias = bias

        self.zeros, self.scales = process_zeros_scales(zeros, scales, bits, self.M)


        if bits == 4:
            if verbose:
                print(self.N // packed, self.M, self.mb // packed, self.tb, self.cutoff)
            self.weight = torch.zeros(int(self.N // packed * self.M)).int().contiguous()
            qinfer.pack4(qweights.int().contiguous(),self.weight, self.N // packed, self.M, self.mb, self.tb, self.cutoff)# * (self.tt//tb))
        elif bits == 3:
            self.weight = torch.zeros(int(self.N // packed * 3 * self.M)).int().contiguous()
            if verbose:
                print(self.N // packed * 3, self.M, self.mb // packed * 3, self.tb, self.cutoff)
            qinfer.pack3(qweights.int().contiguous(),self.weight, self.N // packed * 3, self.M, self.mb // packed * 3, self.tb, self.cutoff)
        elif bits == 2:
            self.weight = torch.zeros(int(self.N // packed * self.M)).int().contiguous()
            qinfer.pack2(qweights.int().contiguous(),self.weight, self.N // packed, self.M, self.mb, self.tb, self.cutoff)# * (self.tt//tb))
                
                
                
    def forward(self, x):
        x = x.reshape((-1, x.shape[-1]))
        B = x.shape[0]
        new_x = x.T.contiguous()
        out = torch.zeros((B, self.M), dtype=torch.float32).contiguous()
        sums = compute_reductions(x,gs=self.gs,cpp=True)
        sums = sums.contiguous()
        if self.gs == -1:
            if self.bits == 4:
                qinfer.forward4(new_x.contiguous(), self.weight.contiguous(), out.contiguous(), self.bias.contiguous(), 
                        self.scales.contiguous(), self.zeros.contiguous(), sums.contiguous(), B, self.N, self.M, B, self.mb, self.tb, self.tt, self.cutoff)
            elif self.bits == 2:
                qinfer.forward2(new_x.contiguous(), self.weight.contiguous(), out.contiguous(), self.bias.contiguous(), 
                        self.scales.contiguous(), self.zeros.contiguous(), sums.contiguous(), B, self.N, self.M, B, self.mb, self.tb, self.tt, self.cutoff)
            elif self.bits == 3:
                qinfer.forward3(new_x.contiguous(), self.weight.contiguous(), out.contiguous(), self.bias.contiguous(), 
                        self.scales.contiguous(), self.zeros.contiguous(), sums.contiguous(), B, self.N, self.M, B, self.mb, self.tb, self.tt, self.cutoff)
        else:
            if self.bits == 4:
                qinfer.forward_gs4(new_x.contiguous(), self.weight.contiguous(), out.contiguous(), self.bias.contiguous(), 
                        self.scales.contiguous(), self.zeros.contiguous(), sums.contiguous(), B, self.N, self.M, B, self.mb, self.tb, self.tt, self.gs, self.cutoff)
            elif self.bits == 2:
                qinfer.forward_gs2(new_x.contiguous(), self.weight.contiguous(), out.contiguous(), self.bias.contiguous(), 
                        self.scales.contiguous(), self.zeros.contiguous(), sums.contiguous(), B, self.N, self.M, B, self.mb, self.tb, self.tt, self.gs, self.cutoff)
            elif self.bits == 3:
                qinfer.forward_gs3(new_x.contiguous(), self.weight.contiguous(), out.contiguous(), self.bias.contiguous(),
                        self.scales.contiguous(), self.zeros.contiguous(), sums.contiguous(), B, self.N, self.M, B, self.mb, self.tb, self.tt, self.gs, self.cutoff)

        return out  

def swap_module(network, module_name, new_module):
    name_parts = module_name.split('.')
    parent = network
    for part in name_parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    
    last_part = name_parts[-1]
    if last_part.isdigit():
        parent[int(last_part)] = new_module
    else:
        setattr(parent, last_part, new_module)

def swap_modules(version, in_network, checkpoint, bits, p, l1, inplace=False, verbose=False, hint=1, qzeros=True, gs=-1, simulate_gs=-1):
    global params
    params = {}

    if version == 'llama':
        preamble = "model"
    elif version == 'opt':
        preamble = "model.decoder"
    else:
        print(f'unknown version {version}')
        return

    if not inplace:
        network = deepcopy(in_network)
    else:
        network = in_network

    if not qzeros:
        zeros = 'zeros'
    else:
        zeros = 'qzeros'
    
    for name, module in network.named_modules():
        is_linear = isinstance(module, torch.nn.Linear)
            
        if not is_linear:
            if verbose:
                print(f'module {name} not replaced')
            continue
        
        try:
            if version == 'llama':
                layer_type = name.split('.')[4]
                module_name = name.split('.')[3]
                index_number= name.split('.')[2]
                bias = None
                start = f"{preamble}.layers.{index_number}.{module_name}.{layer_type}"
            elif version == 'opt':
                layer_type = name.split('.')[-1]
                module_name = name.split('.')[-2]
                index_number= name.split('.')[-3]
                if 'fc' in layer_type:
                    start = f"{preamble}.layers.{module_name}.{layer_type}"
                else:
                    start = f"{preamble}.layers.{index_number}.{module_name}.{layer_type}"
                bias = checkpoint[f"{start}.bias"].float(),

        

            if simulate_gs == -1:
                new_module = qLinear(mode='llama', p=p, l1=l1, name=f"{start}",
                                    zeros=checkpoint[f"{start}.{zeros}"], 
                                    scales = checkpoint[f"{start}.scales"].float(), 
                                    bias = bias,
                                    qweights = checkpoint[f"{start}.qweight"].contiguous(), 
                                    N=module.in_features, M=module.out_features, bits=bits, hint=hint,verbose=verbose,gs=gs)
            else:
                tmp_zeros = checkpoint[f"{start}.{zeros}"]
                tmp_scales = checkpoint[f"{start}.scales"].float()
                if gs != -1:
                    tmp_zeros = tmp_zeros[:,0]
                    tmp_scales = tmp_scales[:,0]
                zeros_tensor = tmp_zeros.repeat(module.in_features//simulate_gs,1)
                scales_tensor = tmp_scales.repeat(module.in_features//simulate_gs,1)
                new_module = qLinear(mode='llama', p=p, l1=l1, name=f"{start}", zeros=zeros_tensor,
                                    scales = scales_tensor, 
                                    bias = bias,
                                    qweights = checkpoint[f"{start}.qweight"].contiguous(), 
                                    N=module.in_features, M=module.out_features, bits=bits, hint=hint,verbose=verbose,gs=simulate_gs)
            
            swap_module(network, name, new_module)

            if verbose:
                print(f'module {name} replaced with {preamble}.layers.{index_number}.{module_name}.{layer_type}')
        except Exception as e:
            if verbose:
                print(e)
                print(f'module {name} not replaced')
        

    return network

