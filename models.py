import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import numpy as np
from torch import einsum
import math
from layers import *
from einops.layers.torch import Reduce
from typing import Tuple, Union
from collections import defaultdict
from einops.layers.torch import Reduce
import operator 


from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *


import torch
from torch import nn
import numpy as np

from scipy import stats


class UNet(nn.Module):

   def __init__(self, args):
        super().__init__()
        #if len (enc_blocks) != len(dec_blocks) + 1:
        #   raise Exception('length of list of encoder blocks should be 1 greater than the list of decoder blocks')
        self.encoder = Encoder(channels=args.channels, in_ch = args.in_ch,groups=args.groups, blocks=args.enc_blocks, dim=args.dim, act= args.act,  dropout=args.dropout)
        self.dec_channels = args.channels * (2**len(args.enc_blocks))
        self.decoder = Decoder(channels=self.dec_channels, out_ch = args.out_ch, groups=args.groups, blocks=args.dec_blocks, dim=args.dim, act= args.act,  with_conv = args.with_conv, dropout=args.dropout)

   def forward(self, x):
        out_e = self.encoder(x)
        out = self.decoder(out_e)
        return out

class CrossUNet(nn.Module):

   def __init__(self, args):
        super().__init__()
        #if len (enc_blocks) != len(dec_blocks) + 1:
        #   raise Exception('length of list of encoder blocks should be 1 greater than the list of decoder blocks')
        self.encoder = CrossEncoder(channels=args.channels, in_ch = args.in_ch,groups=args.groups, blocks=args.enc_blocks, dim=args.dim, act= args.act,  dropout=args.dropout)
        self.dec_channels = args.channels * (2**len(args.enc_blocks))
        self.decoder = CrossDecoder(channels=self.dec_channels, out_ch = args.out_ch, groups=args.groups, blocks=args.dec_blocks, dim=args.dim, act= args.act,  with_conv = args.with_conv, dropout=args.dropout)

   def forward(self, x):
        q_loss = 0
        out_e = self.encoder(x)
        out = self.decoder(out_e)

        return out, q_loss


class ShapeVQUNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        if len (args.enc_blocks) != len(args.dec_blocks) + 1:
            raise Exception('length of list of encoder blocks should be 1 greater than the list of decoder blocks')
        self.encoder = CrossEncoder(channels=args.channels,in_ch = args.in_ch, groups=args.groups, blocks=args.enc_blocks, dim=args.dim, act= args.act,  dropout=args.dropout)
        self.dec_channels = args.channels * (2**len(args.enc_blocks))
        self.VQ = args.VQ
        self.quant_dim = args.embed_dim if args.quantise == 'spatial' else int(math.prod([args.image_size[i]//(2**(len(args.enc_blocks)-1)) for i in range(len(args.image_size))]))
        self.pre_quant_conv = torch.nn.Conv2d(self.dec_channels, args.embed_dim, kernel_size=1,stride=1, padding=0) if args.dim == '2D' else torch.nn.Conv3d(self.dec_channels, args.embed_dim, kernel_size=1,stride=1, padding=0)
        self.quantise = VectorQuantiser(n_e = args.n_e, e_dim = self.quant_dim, quantise = args.quantise, dim = args.dim, beta=0.25) if self.VQ == True else nn.Identity()
        self.n = torch.nn.InstanceNorm3d(args.embed_dim) if args.dim =='3D' else torch.nn.InstanceNorm2d(args.embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(args.embed_dim, self.dec_channels, kernel_size=1,stride=1, padding=0) if args.dim == '2D' else torch.nn.Conv3d(args.embed_dim, self.dec_channels, kernel_size=1,stride=1, padding=0)
        self.midblock = ResnetBlock(in_channels=self.dec_channels, out_channels=self.dec_channels,dim = args.dim,  groups = args.groups, act = args.act, dropout = args.dropout)
        self.decoder = CrossDecoder(channels=self.dec_channels, out_ch = args.out_ch,  groups=args.groups, blocks=args.dec_blocks, dim=args.dim, act= args.act,  with_conv = args.with_conv, dropout=args.dropout)

    
    def forward(self, x):
        t = torch.unsqueeze(x[:,0], dim =1)
        out_t = self.encoder(t)
        out_tp = self.pre_quant_conv(out_t[-1])
        if x.shape[1] == 2:
           a = torch.unsqueeze(x[:,1], dim =1)
           out_a = self.encoder(a)
        else:
           out_a = None
        out_a = out_a[-1] if out_a != None else out_a
        if self.VQ == True:
           vqts, q_lossts, infots = self.quantise(out_tp)
           
        else:
           vqts, q_lossts, infots = out_tp, 0, 0
        
        vq_post = self.post_quant_conv(vqts)
        vq_post = self.midblock(vq_post)
        dec_in = out_t[:-1]
        dec_in.append(vq_post)
        outs = self.decoder(dec_in)
        return outs, out_t[-1], out_a, vqts, q_lossts
    
class ShapeVQUNetHybrid(nn.Module):

    def __init__(self, args):
        super().__init__()
        if len (args.enc_blocks) != len(args.dec_blocks) + 1:
            raise Exception('length of list of encoder blocks should be 1 greater than the list of decoder blocks')
        self.encoder = CrossEncoderH(channels=args.channels,in_ch = args.in_ch,TD = args.ch_3D, groups=args.groups, blocks=args.enc_blocks, dim=args.dim, act= args.act,  dropout=args.dropout)
        self.dec_channels = args.channels * (2**len(args.enc_blocks))
        self.VQ = args.VQ
        self.quant_dim = args.embed_dim if args.quantise == 'spatial' else int(math.prod([args.image_size[i]//(2**(len(args.enc_blocks)-1)) for i in range(len(args.image_size))]))
        self.pre_quant_conv = torch.nn.Conv2d(self.dec_channels, args.embed_dim, kernel_size=1,stride=1, padding=0) if args.dim == '2D' else torch.nn.Conv3d(self.dec_channels, args.embed_dim, kernel_size=1,stride=1, padding=0)
        self.quantise = VectorQuantiser(n_e = args.n_e, e_dim = self.quant_dim, quantise = args.quantise, dim = args.dim, beta=0.25) if self.VQ == True else nn.Identity()
        self.n = torch.nn.InstanceNorm3d(args.embed_dim) if args.dim =='3D' else torch.nn.InstanceNorm2d(args.embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(args.embed_dim, self.dec_channels, kernel_size=1,stride=1, padding=0) if args.dim == '2D' else torch.nn.Conv3d(args.embed_dim, self.dec_channels, kernel_size=1,stride=1, padding=0)
        self.midblock = ResnetBlock(in_channels=self.dec_channels, out_channels=self.dec_channels,dim = args.dim,  groups = args.groups, act = args.act, dropout = args.dropout)
        self.decoder = CrossDecoderH(channels=self.dec_channels, out_ch = args.out_ch, TD = args.ch_3D,  groups=args.groups, blocks=args.dec_blocks, dim=args.dim, act= args.act,  with_conv = args.with_conv, dropout=args.dropout)

    
    def forward(self, x):
        t = torch.unsqueeze(x[:,0], dim =1)
        out_t = self.encoder(t)
        out_tp = self.pre_quant_conv(out_t[-1])
        if x.shape[1] == 2:
           a = torch.unsqueeze(x[:,1], dim =1)
           a = torch.unsqueeze(x[:,1], dim =1)
           out_a = self.encoder(a)
        else:
           out_a = None
        out_a = out_a[-1] if out_a != None else out_a
        if self.VQ == True:
           vqts, q_lossts, infots = self.quantise(out_tp)
           
        else:
           vqts, q_lossts, infots = out_tp, 0, 0
        

        vq_post = self.post_quant_conv(vqts)
        vq_post = self.midblock(vq_post)

        dec_in = out_t[:-1]
        dec_in.append(vq_post)
        outs = self.decoder(dec_in)
        return outs, out_t[-1], out_a, vqts, q_lossts

class SE3AE(nn.Module):

    def __init__(self,args):

        super(SE3AE, self).__init__()

        self.gs = rot3dOnR3()

        self.num_features = args.num_features
        self.multiplicity = args.multiplicity
        self.so3channels = args.so3channels

        self.in_type = FieldType(self.gs, [self.gs.trivial_repr])

        self._init = args.init
        layer_types = []
        layer_types2 = []
        for i in range(len(self.multiplicity)):
            layer_types.append((FieldType(self.gs, [self.build_representation(self.num_features[i])] * self.multiplicity[i]), self.so3channels[i]))

        blocks = [
            R3Conv(self.in_type, layer_types[0][0], kernel_size=5, padding=2, bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                REqResBlock3D(layer_types[i][0], layer_types[i][1], layer_types[i+1][0], 2, features=args.res_features)
            )

        for i in list(reversed(range(len(self.multiplicity)))):
            layer_types2.append((FieldType(self.gs, [self.build_representation(self.num_features[i])] * self.multiplicity[i]), self.so3channels[i]))
        
        blocks2 = [R3Conv(layer_types2[0][0], layer_types2[0][0], kernel_size=3, padding=1, bias=False, initialize=False)]
        blocks2.append(R3ConvTransposed(layer_types2[0][0], layer_types2[0][0], kernel_size=3,stride = 2, padding =1, output_padding = 1, bias=False, initialize=False))
        
        for i in range(len(layer_types2) - 1):
            blocks2.append(
                EqResBlock3D(layer_types2[i][0], layer_types2[i][1], layer_types2[i+1][0], 1, features=args.res_features)
            )
            blocks2.append(R3ConvTransposed(layer_types2[i+1][0], layer_types2[i+1][0], kernel_size=3, stride = 2, padding =1, output_padding = 1, bias=False, initialize=False)) if  i < len(layer_types) - 2 else None
        
        
        blocks2.append(
            R3Conv(layer_types[0][0], self.in_type, kernel_size=5, padding=2, bias=False, initialize=False)
        )
               
        self.blocks = SequentialModule(*blocks)
        self.blocks2 = SequentialModule(*blocks2)


    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, R3Conv):
                if self._init == 'he':
                    init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
                elif self._init == 'rand':
                    m.weights.data[:] = torch.randn_like(m.weights)
                else:
                    raise ValueError()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                m.weight.data[:] = torch.tensor(stats.ortho_group.rvs(max(i, o))[:o, :i])
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_representation(self, K: int):
        assert K >= 0

        if K == 0:
            return [self.gs.trivial_repr]

        SO3 = self.gs.fibergroup

        polinomials = [self.gs.trivial_repr, SO3.irrep(1)]

        for k in range(2, K+1):
            polinomials.append(
                polinomials[-1].tensor(SO3.irrep(1))
            )

        return directsum(polinomials, name=f'polynomial_{K}')

    def forward(self, input: torch.Tensor):

        input= GeometricTensor(input, self.in_type)

        features = self.blocks(input)
        output = self.blocks2(features)
        return output

class VQSE3AE(nn.Module):

    def __init__(self,args):

        super(VQSE3AE, self).__init__()

        self.gs = rot3dOnR3()

        self.num_features = args.num_features
        self.multiplicity = args.multiplicity
        self.so3channels = args.so3channels

        self.in_type = FieldType(self.gs, [self.gs.trivial_repr])

        self._init = args.init
        self.VQ = args.VQ
        layer_types = []
        layer_types2 = []
        for i in range(len(self.multiplicity)):
            layer_types.append((FieldType(self.gs, [self.build_representation(self.num_features[i])] * self.multiplicity[i]), self.so3channels[i]))

        blocks = [
            R3Conv(self.in_type, layer_types[0][0], kernel_size=5, padding=2, bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                REqResBlock3D(layer_types[i][0], layer_types[i][1], layer_types[i+1][0], 2, features=args.res_features)
            )

        for i in list(reversed(range(len(self.multiplicity)))):
            layer_types2.append((FieldType(self.gs, [self.build_representation(self.num_features[i])] * self.multiplicity[i]), self.so3channels[i]))
        self.quantise = VectorQuantiser(n_e = args.n_e, e_dim = self.quant_dim, quantise = args.quantise, dim = args.dim, beta=0.25) if self.VQ == True else nn.Identity()
        blocks2 = [R3Conv(layer_types2[0][0], layer_types2[0][0], kernel_size=3, padding=1, bias=False, initialize=False)]
        blocks2.append(R3ConvTransposed(layer_types2[0][0], layer_types2[0][0], kernel_size=3,stride = 2, padding =1, output_padding = 1, bias=False, initialize=False))
        
        for i in range(len(layer_types2) - 1):
            blocks2.append(
                EqResBlock3D(layer_types2[i][0], layer_types2[i][1], layer_types2[i+1][0], 1, features=args.res_features)
            )
            blocks2.append(R3ConvTransposed(layer_types2[i+1][0], layer_types2[i+1][0], kernel_size=3, stride = 2, padding =1, output_padding = 1, bias=False, initialize=False)) if  i < len(layer_types) - 2 else None
        
        
        blocks2.append(
            R3Conv(layer_types[0][0], self.in_type, kernel_size=5, padding=2, bias=False, initialize=False)
        )
               
        self.blocks = SequentialModule(*blocks)
        self.blocks2 = SequentialModule(*blocks2)


    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, R3Conv):
                if self._init == 'he':
                    init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
                elif self._init == 'rand':
                    m.weights.data[:] = torch.randn_like(m.weights)
                else:
                    raise ValueError()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                m.weight.data[:] = torch.tensor(stats.ortho_group.rvs(max(i, o))[:o, :i])
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_representation(self, K: int):
        assert K >= 0

        if K == 0:
            return [self.gs.trivial_repr]

        SO3 = self.gs.fibergroup

        polinomials = [self.gs.trivial_repr, SO3.irrep(1)]

        for k in range(2, K+1):
            polinomials.append(
                polinomials[-1].tensor(SO3.irrep(1))
            )

        return directsum(polinomials, name=f'polynomial_{K}')

    def forward(self, input):
        t = torch.unsqueeze(input[:,0], dim =1)

        inputt2 = GeometricTensor(t, self.in_type)

        featurest2 = self.blocks(inputt2)
        if input.shape[2] == 2:
           a = torch.unsqueeze(input[:,1], dim =1)
           inputadc = GeometricTensor(a, self.in_type)
           featuresadc = self.blocks(inputadc)
        else:
           featuresadc = None

        if self.VQ == True:
           vqts, q_lossts, infots = self.quantise(featurest2)           
        else:
           vqts, q_lossts, infots = out_tp, 0, 0

        output = self.blocks2(vqts)
        return output, featurest2, featuresadc, vqts, q_lossts

class SE3UNET(nn.Module):

    def __init__(self, args):

        super(SE3CNN, self).__init__()

        self.gs = rot3dOnR3()

        self.in_type = FieldType(self.gs, [self.gs.trivial_repr])
        self._init = args.init

        
        self.num_features = args.num_features
        self.multiplicity = args.multiplicity
        self.so3channels = args.se3channels
        
        layer_types = []
        layer_types2 = []
        for i in range(len(self.multiplicity)):
            layer_types.append((FieldType(self.gs, [self.build_representation(self.num_features[i])] * self.multiplicity[i]), self.so3channels[i]))


        blocks = [
            R3Conv(self.in_type, layer_types[0][0], kernel_size=3, padding=1, bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                ResBlock(layer_types[i][0], layer_types[i][1], layer_types[i+1][0], 2, features=res_features)
            )


        for i in list(reversed(range(1,len(self.multiplicity)))):
            j = i-1 if i-1 > 0 else 0
            print(j)
            if i > 1:
               layer_types2.append((FieldType(self.gs, [self.build_representation(self.num_features[j])] * (self.multiplicity[i]+self.multiplicity[j])), self.so3channels[j]))
               layer_types2.append((FieldType(self.gs, [self.build_representation(self.num_features[j])] * (self.multiplicity[j])), self.so3channels[j]))
            else:
               layer_types2.append((FieldType(self.gs, [self.build_representation(self.num_features[j])] * (self.multiplicity[j])), self.so3channels[j]))
  
        
        blocks2 = [R3Conv(layer_types[-1][0], layer_types[-1][0], kernel_size=3, padding=1, bias=False, initialize=False)]
        blocks2.append(R3ConvTransposed(layer_types[-1][0], layer_types[-1][0], kernel_size=3,stride = 2, padding =1, output_padding = 1, bias=False, initialize=False))
     
        
        for i in range((len(layer_types2)+1)//2):
            if i < ((len(layer_types2)+1)//2)-1:
               blocks2.append(ResBlock(layer_types2[2*i][0], layer_types2[2*i][1], layer_types2[2*i+1][0], 1, features=res_features)
               )
               blocks2.append(R3ConvTransposed(layer_types2[2*i+1][0], layer_types2[2*i+1][0], kernel_size=3, stride = 2, padding =1, output_padding = 1, bias=False, initialize=False)) if  i < ((len(layer_types2)+1)//2 )-1 else None 
            else:
               blocks2.append(ResBlock(layer_types2[2*i-1][0], layer_types2[2*i-1][1], layer_types[0][0], 1, features=res_features)
               )

        blocks2.append(
            R3Conv(layer_types[0][0], self.in_type, kernel_size=5, padding=2, bias=False, initialize=False)
        )
        
        
        
        self.blocks = SequentialModule(*blocks)
        self.blocks2 = blocks2


    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, R3Conv):
                if self._init == 'he':
                    init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
                elif self._init == 'rand':
                    m.weights.data[:] = torch.randn_like(m.weights)
                else:
                    raise ValueError()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                m.weight.data[:] = torch.tensor(stats.ortho_group.rvs(max(i, o))[:o, :i])
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_representation(self, K: int):
        assert K >= 0

        if K == 0:
            return [self.gs.trivial_repr]

        SO3 = self.gs.fibergroup

        polinomials = [self.gs.trivial_repr, SO3.irrep(1)]

        for k in range(2, K+1):
            polinomials.append(
                polinomials[-1].tensor(SO3.irrep(1))
            )

        return directsum(polinomials, name=f'polynomial_{K}')

    def forward(self, input: torch.Tensor):
        encouts = []
        input = GeometricTensor(input, self.in_type)
        for l, level in enumerate(self.blocks):
            input = level(input)
            encouts.append(input)
        encouts = list(reversed(encouts))
        dec = encouts[0]
        for j, level in enumerate(self.blocks2):

            if (j % 2 == 1 or j < 2):
               dec = level(dec)
            else:
               dec = tensor_directsum([dec, encouts[j//2]]) if j < (len(self.blocks2)-2) else dec
               dec = level(dec)

        return dec


class VQSE3UNET(nn.Module):

    def __init__(self, args):

        super(VQSE3UNET, self).__init__()

        self.gs = rot3dOnR3()

        self.in_type = FieldType(self.gs, [self.gs.trivial_repr])
        self._init = args.init

        
        self.num_features = args.num_features
        self.multiplicity = args.multiplicity
        self.so3channels = args.se3channels
        self.quantise = VectorQuantiser(n_e = args.n_e, e_dim = self.quant_dim, quantise = args.quantise, dim = args.dim, beta=0.25) if self.VQ == True else nn.Identity()
        self.VQ = args.VQ

        layer_types = []
        layer_types2 = []
        for i in range(len(self.multiplicity)):
            layer_types.append((FieldType(self.gs, [self.build_representation(self.num_features[i])] * self.multiplicity[i]), self.so3channels[i]))


        blocks = [
            R3Conv(self.in_type, layer_types[0][0], kernel_size=3, padding=1, bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                ResBlock(layer_types[i][0], layer_types[i][1], layer_types[i+1][0], 2, features=res_features)
            )

        
        for i in list(reversed(range(1,len(self.multiplicity)))):
            j = i-1 if i-1 > 0 else 0
            if i > 1:
               layer_types2.append((FieldType(self.gs, [self.build_representation(self.num_features[j])] * (self.multiplicity[i]+self.multiplicity[j])), self.so3channels[j]))
               layer_types2.append((FieldType(self.gs, [self.build_representation(self.num_features[j])] * (self.multiplicity[j])), self.so3channels[j]))
            else:
               layer_types2.append((FieldType(self.gs, [self.build_representation(self.num_features[j])] * (self.multiplicity[j])), self.so3channels[j]))
  
        
        blocks2 = [R3Conv(layer_types[-1][0], layer_types[-1][0], kernel_size=3, padding=1, bias=False, initialize=False)]
        blocks2.append(R3ConvTransposed(layer_types[-1][0], layer_types[-1][0], kernel_size=3,stride = 2, padding =1, output_padding = 1, bias=False, initialize=False))
     
        
        for i in range((len(layer_types2)+1)//2):
            if i < ((len(layer_types2)+1)//2)-1:
               blocks2.append(ResBlock(layer_types2[2*i][0], layer_types2[2*i][1], layer_types2[2*i+1][0], 1, features=res_features)
               )
               blocks2.append(R3ConvTransposed(layer_types2[2*i+1][0], layer_types2[2*i+1][0], kernel_size=3, stride = 2, padding =1, output_padding = 1, bias=False, initialize=False)) if  i < ((len(layer_types2)+1)//2 )-1 else None 
            else:
               blocks2.append(ResBlock(layer_types2[2*i-1][0], layer_types2[2*i-1][1], layer_types[0][0], 1, features=res_features)
               )

        blocks2.append(
            R3Conv(layer_types[0][0], self.in_type, kernel_size=5, padding=2, bias=False, initialize=False)
        )
        
        
        
        self.blocks = SequentialModule(*blocks)
        self.blocks2 = blocks2


    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, R3Conv):
                if self._init == 'he':
                    init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
                elif self._init == 'rand':
                    m.weights.data[:] = torch.randn_like(m.weights)
                else:
                    raise ValueError()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                m.weight.data[:] = torch.tensor(stats.ortho_group.rvs(max(i, o))[:o, :i])
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_representation(self, K: int):
        assert K >= 0

        if K == 0:
            return [self.gs.trivial_repr]

        SO3 = self.gs.fibergroup

        polinomials = [self.gs.trivial_repr, SO3.irrep(1)]

        for k in range(2, K+1):
            polinomials.append(
                polinomials[-1].tensor(SO3.irrep(1))
            )

        return directsum(polinomials, name=f'polynomial_{K}')

    def forward(self, input: torch.Tensor):
        encoutst2 = []
        t = torch.unsqueeze(input[:,0], dim =1)

        t2 = GeometricTensor(t, self.in_type)
        
        input = GeometricTensor(input, self.in_type)
        for l, level in enumerate(self.blocks):
            t2 = level(t2)
            encoutst2.append(t2)
        encoutst2 = list(reversed(encouts))
        if input.shape[2] == 2:
           a = torch.unsqueeze(input[:,1], dim =1)
           adc = GeometricTensor(a, self.in_type)
           adc = self.blocks(adc)
        else:
           adc = None
        dec = encoutst2[0]
        if self.VQ == True:
           vqts, q_lossts, infots = self.quantise(dec)           
        else:
           vqts, q_lossts, infots = out_tp, 0, 0
        for j, level in enumerate(self.blocks2):

            if (j % 2 == 1 or j < 2):
               dec = level(dec)
            else:
               dec = tensor_directsum([dec, encouts[j//2]]) if j < (len(self.blocks2)-2) else dec
               dec = level(dec)

        return dec, encoutst2[0], adc, vqts, q_lossts

class SE2CNN(nn.Module):

    def __init__(self, args):

        super(SE2CNN, self).__init__()

        self.gs = rot2dOnR2(N=args.group)

        self.in_type = FieldType(self.gs, [self.gs.trivial_repr])
        self._init = args.init
        self.multiplicity = args.multiplicity
        if args.repr == 'Regular':
           field = self.gs.regular_repr
        else:
           field = self.gs.irrep(1)
        layer_types = []
        layer_types2 = []
        for i in range(len(self.multiplicity)+1):
            if i == 0:
               layer_types = [FieldType(self.gs, args.in_ch*[self.gs.trivial_repr])]
            else:
               layer_types.append(FieldType(self.gs, self.multiplicity[i-1]*[field]))


        blocks = [
            R2Conv(layer_types[0], layer_types[1], kernel_size=3, padding=1, bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 2):
            blocks.append(
               EqResBlock2D(layer_types[i+1], layer_types[i+2], args.repr, 2)
            )
        
        for i in list(reversed(range(1,len(layer_types)))):
            layer_types2.append(FieldType(self.gs, (self.multiplicity[i]+self.multiplicity[i-1])*[field]))
            layer_types2.append(FieldType(self.gs, (self.multiplicity[i-1])*[field]))

        blocks2 = [EqResBlock2D(layer_types[-1], layer_types[-1], args.repr, 1)]
        blocks2.append(R2ConvTransposed(layer_types[-1], layer_types[-1], kernel_size=3,stride = 2, padding =1, output_padding = 1, bias=False, initialize=False))
        
        for i in range((len(layer_types2))//2):
            if i < ((len(layer_types2)+1)//2)-1:
               blocks2.append(EqResBlock2D(layer_types2[2*i], layer_types2[2*i+1], args.repr, 1)
               )
               blocks2.append(R2ConvTransposed(layer_types2[2*i+1], layer_types2[2*i+1], kernel_size=3, stride = 2, padding =1, output_padding = 1, bias=False, initialize=False)) if  i < (len(layer_types2)//2)-2 else None 
    
        blocks2.append(
            R2Conv(layer_types[1], layer_types[0], kernel_size=3, padding=1, bias=False, initialize=False)
        )
                
        self.blocks = SequentialModule(*blocks)
        self.blocks2 = blocks2


    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, R3Conv):
                if self._init == 'he':
                    init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
                elif self._init == 'rand':
                    m.weights.data[:] = torch.randn_like(m.weights)
                else:
                    raise ValueError()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                m.weight.data[:] = torch.tensor(stats.ortho_group.rvs(max(i, o))[:o, :i])
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor):
        encouts = []
        input = GeometricTensor(input, self.in_type)
        for l, level in enumerate(self.blocks):
            input = level(input)
            encouts.append(input)
        encouts = list(reversed(encouts))
        dec = encouts[0]
        
        for j, level in enumerate(self.blocks2):
            if (j % 2 == 1 or j < 2):
               dec = level(dec)
            else:
               dec = tensor_directsum([dec, encouts[j//2]]) if j < (len(self.blocks2)-1) else dec
               dec = level(dec)        
        return dec

class VQSE2CNN(nn.Module):

    def __init__(self, args):

        super(VQSE2CNN, self).__init__()

        self.gs = rot2dOnR2(N=args.group)

        self.in_type = FieldType(self.gs, [self.gs.trivial_repr])
        self._init = args.init
        self.multiplicity = args.multiplicity
        self.quantise = VectorQuantiser(n_e = args.n_e, e_dim = self.quant_dim, quantise = args.quantise, dim = args.dim, beta=0.25) if self.VQ == True else nn.Identity()
        self.VQ = args.VQ

        if args.repr == 'Regular':
           field = self.gs.regular_repr
        else:
           field = self.gs.irrep(1)
        layer_types = []
        layer_types2 = []
        for i in range(len(self.multiplicity)+1):
            if i == 0:
               layer_types = [FieldType(self.gs, args.in_ch*[self.gs.trivial_repr])]
            else:
               layer_types.append(FieldType(self.gs, self.multiplicity[i-1]*[field]))


        blocks = [
            R2Conv(layer_types[0], layer_types[1], kernel_size=3, padding=1, bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 2):
            blocks.append(
               EqResBlock2D(layer_types[i+1], layer_types[i+2], args.repr, 2)
            )
        
        for i in list(reversed(range(1,len(layer_types)))):
            layer_types2.append(FieldType(self.gs, (self.multiplicity[i]+self.multiplicity[i-1])*[field]))
            layer_types2.append(FieldType(self.gs, (self.multiplicity[i-1])*[field]))

        blocks2 = [EqResBlock2D(layer_types[-1], layer_types[-1], args.repr, 1)]
        blocks2.append(R2ConvTransposed(layer_types[-1], layer_types[-1], kernel_size=3,stride = 2, padding =1, output_padding = 1, bias=False, initialize=False))
        
        for i in range((len(layer_types2))//2):
            if i < ((len(layer_types2)+1)//2)-1:
               blocks2.append(EqResBlock2D(layer_types2[2*i], layer_types2[2*i+1], args.repr, 1)
               )
               blocks2.append(R2ConvTransposed(layer_types2[2*i+1], layer_types2[2*i+1], kernel_size=3, stride = 2, padding =1, output_padding = 1, bias=False, initialize=False)) if  i < (len(layer_types2)//2)-2 else None 
    
        blocks2.append(
            R2Conv(layer_types[1], layer_types[0], kernel_size=3, padding=1, bias=False, initialize=False)
        )
                
        self.blocks = SequentialModule(*blocks)
        self.blocks2 = blocks2


    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, R3Conv):
                if self._init == 'he':
                    init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
                elif self._init == 'rand':
                    m.weights.data[:] = torch.randn_like(m.weights)
                else:
                    raise ValueError()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                m.weight.data[:] = torch.tensor(stats.ortho_group.rvs(max(i, o))[:o, :i])
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor):
        encoutst2 = []
        t = torch.unsqueeze(input[:,0], dim =1)

        t2 = GeometricTensor(t, self.in_type)

        for l, level in enumerate(self.blocks):
            t2 = level(t2)
            encoutst2.append(t2)
        encoutst2 = list(reversed(encoutst2))
        dec = encoutst2[0]
        if input.shape[2] == 2:
           a = torch.unsqueeze(input[:,1], dim =1)
           inputadc = GeometricTensor(a, self.in_type)
           adc = self.blocks(inputadc)
        else:
           adc = None
        if self.VQ == True:
           vqts, q_lossts, infots = self.quantise(dec)           
        else:
           vqts, q_lossts, infots = out_tp, 0, 0
        
        for j, level in enumerate(self.blocks2):
            if (j % 2 == 1 or j < 2):
               dec = level(dec)
            else:
               dec = tensor_directsum([dec, encouts[j//2]]) if j < (len(self.blocks2)-1) else dec
               dec = level(dec)        
        return dec, encoutst2[0], adc, vqts, q_lossts
    
class HybridSECNN(nn.Module):

    def __init__(self, args):

        super(HybridSECNN, self).__init__()

        self.gs = rot2dOnR2(N=args.group)
        self.gs3D = rot3dOnR3()

        self.in_type = FieldType(self.gs, [self.gs.trivial_repr])
        
        self._init = args.init
        
        self.multiplicity = args.multiplicity
        self.multiplicity3D = args.multiplicity3D
        if args.repr == 'Regular':
           field = self.gs.regular_repr
        else:
           field = self.gs.irrep(1)
        layer_types = []
        layer_types2 = []
        for i in range(len(self.multiplicity)):
            if i == 0:
               layer_types.append(FieldType(self.gs, args.in_ch*[self.gs.trivial_repr]))      
            else:
               layer_types.append(FieldType(self.gs, self.multiplicity[i]*[field]))
        self.latent_type3d = FieldType(self.gs3D, group * self.multiplicity[-1] * [self.gs3D.trivial_repr])
        self.latent_type2d = FieldType(self.gs, group * self.multiplicity[-1] * [self.gs.trivial_repr])
        blocktypes = [FieldType(self.gs3D, group * self.multiplicity[-1] * [self.gs3D.trivial_repr]),
                       (FieldType(self.gs3D, [self.build_representation(3)] * self.multiplicity3D[-1]), 960)]
        blocks = [
            R2Conv(layer_types[0], layer_types[1], kernel_size=3, padding=1, bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 2):
            blocks.append(
               EqResBlock2D(layer_types[i+1], layer_types[i+2], args.repr, 2)
            )
        self.blockslatent1 = ResBlock(blocktypes[0], blocktypes[1][1], blocktypes[1][0], 1, features=args.res_features)
        self.blockslatent2 = ResBlock(blocktypes[1][0], blocktypes[1][1], blocktypes[0], 1, features=args.res_features)
        
        for i in list(reversed(range(1,len(layer_types)))):
            layer_types2.append(FieldType(self.gs, (self.multiplicity[i]+self.multiplicity[i-1])*[field]))
            layer_types2.append(FieldType(self.gs, (self.multiplicity[i-1])*[field]))
        blocks2 = [R2ConvTransposed(FieldType(self.gs, group * self.multiplicity[-1] * [self.gs.trivial_repr]), layer_types[-1], kernel_size=3,stride = 2, padding =1, output_padding = 1, bias=False, initialize=False)]

        
        for i in range((len(layer_types2))//2):
            blocks2.append(EqResBlock2D(layer_types2[2*i], layer_types2[2*i+1], args.repr, 1)
               )
            blocks2.append(R2ConvTransposed(layer_types2[2*i+1], layer_types2[2*i+1], kernel_size=3, stride = 2, padding =1, output_padding = 1, bias=False, initialize=False)) if  i < (len(layer_types2)//2)-2 else None 
                   
        blocks2.append(
            R2Conv(layer_types[1], layer_types[0], kernel_size=3, padding=1, bias=False, initialize=False)
        )
               
        self.blocks = blocks
        self.blocks2 = blocks2



    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, R3Conv):
                if self._init == 'he':
                    init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
                elif self._init == 'rand':
                    m.weights.data[:] = torch.randn_like(m.weights)
                else:
                    raise ValueError()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                m.weight.data[:] = torch.tensor(stats.ortho_group.rvs(max(i, o))[:o, :i])
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_representation(self, K: int):
        assert K >= 0

        if K == 0:
            return [self.gs3D.trivial_repr]

        SO3 = self.gs3D.fibergroup

        polinomials = [self.gs3D.trivial_repr, SO3.irrep(1)]

        for k in range(2, K+1):
            polinomials.append(
                polinomials[-1].tensor(SO3.irrep(1))
            )

        return directsum(polinomials, name=f'polynomial_{K}')

    def forward(self, input: torch.Tensor):
        encoutsdic = {}
        output_tensor = []
        for m in range(len(self.blocks)):
            encoutsdic[f'encout{m}'] = []
        z = input.shape[-1]
        for i in range(input.shape[-1]):
            inputx = input[:,:,:,:,i]
            inputx = GeometricTensor(inputx, self.in_type)
            for l, level in enumerate(self.blocks):
                inputx = level(inputx)
                encoutsdic[f'encout{l}'].append(inputx)
        latent = torch.cat([torch.unsqueeze(t.tensor, dim = 4) for t in encoutsdic[f'encout{len(self.blocks)-1}']], dim = 4)
    
        latent = GeometricTensor(latent, self.latent_type3d)
        latent = self.blockslatent1(latent)
        latent = self.blockslatent2(latent)
        for i in range(input.shape[-1]):
            dec = torch.squeeze(latent[:,:,:,:,i].tensor)
            dec = GeometricTensor(dec, self.latent_type2d)
            for j, level in enumerate(self.blocks2):      
                if (j % 2 == 0):
                   dec = level(dec) if j < (len(self.blocks2)-2) else dec
                else:
                   if (len(self.blocks)-((j+1)//2)+1) > 1:
                       dec = tensor_directsum([dec, encoutsdic[f'encout{len(self.blocks)-((j+1)//2+1)}'][i]]) 
                   dec = level(dec)
            output_tensor.append(dec)
    
        out = torch.cat([torch.unsqueeze(t.tensor, dim = 4) for t in output_tensor], dim = 4)
        
        
        
        return out

class VQHybridSECNN(nn.Module):

    def __init__(self, args):

        super(VQHybridSECNN, self).__init__()

        self.gs = rot2dOnR2(N=args.group)
        self.gs3D = rot3dOnR3()

        self.in_type = FieldType(self.gs, [self.gs.trivial_repr])
        
        self._init = args.init
        
        self.multiplicity = args.multiplicity
        self.multiplicity3D = args.multiplicity3D
        if args.repr == 'Regular':
           field = self.gs.regular_repr
        else:
           field = self.gs.irrep(1)
        layer_types = []
        layer_types2 = []
        for i in range(len(self.multiplicity)):
            if i == 0:
               layer_types.append(FieldType(self.gs, args.in_ch*[self.gs.trivial_repr]))      
            else:
               layer_types.append(FieldType(self.gs, self.multiplicity[i]*[field]))
        self.latent_type3d = FieldType(self.gs3D, group * self.multiplicity[-1] * [self.gs3D.trivial_repr])
        self.latent_type2d = FieldType(self.gs, group * self.multiplicity[-1] * [self.gs.trivial_repr])
        blocktypes = [FieldType(self.gs3D, group * self.multiplicity[-1] * [self.gs3D.trivial_repr]),
                       (FieldType(self.gs3D, [self.build_representation(3)] * self.multiplicity3D[-1]), 960)]
        blocks = [
            R2Conv(layer_types[0], layer_types[1], kernel_size=3, padding=1, bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 2):
            blocks.append(
               EqResBlock2D(layer_types[i+1], layer_types[i+2], args.repr, 2)
            )
        self.blockslatent1 = ResBlock(blocktypes[0], blocktypes[1][1], blocktypes[1][0], 1, features=args.res_features)
        self.blockslatent2 = ResBlock(blocktypes[1][0], blocktypes[1][1], blocktypes[0], 1, features=args.res_features)
        self.quantise = VectorQuantiser(n_e = args.n_e, e_dim = self.quant_dim, quantise = args.quantise, dim = args.dim, beta=0.25) if self.VQ == True else nn.Identity()

        for i in list(reversed(range(1,len(layer_types)))):
            layer_types2.append(FieldType(self.gs, (self.multiplicity[i]+self.multiplicity[i-1])*[field]))
            layer_types2.append(FieldType(self.gs, (self.multiplicity[i-1])*[field]))
        blocks2 = [R2ConvTransposed(FieldType(self.gs, group * self.multiplicity[-1] * [self.gs.trivial_repr]), layer_types[-1], kernel_size=3,stride = 2, padding =1, output_padding = 1, bias=False, initialize=False)]

        
        for i in range((len(layer_types2))//2):
            blocks2.append(EqResBlock2D(layer_types2[2*i], layer_types2[2*i+1], args.repr, 1)
               )
            blocks2.append(R2ConvTransposed(layer_types2[2*i+1], layer_types2[2*i+1], kernel_size=3, stride = 2, padding =1, output_padding = 1, bias=False, initialize=False)) if  i < (len(layer_types2)//2)-2 else None 
                   
        blocks2.append(
            R2Conv(layer_types[1], layer_types[0], kernel_size=3, padding=1, bias=False, initialize=False)
        )
               
        self.blocks = blocks
        self.blocks2 = blocks2



    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, R3Conv):
                if self._init == 'he':
                    init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
                elif self._init == 'rand':
                    m.weights.data[:] = torch.randn_like(m.weights)
                else:
                    raise ValueError()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                m.weight.data[:] = torch.tensor(stats.ortho_group.rvs(max(i, o))[:o, :i])
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_representation(self, K: int):
        assert K >= 0

        if K == 0:
            return [self.gs3D.trivial_repr]

        SO3 = self.gs3D.fibergroup

        polinomials = [self.gs3D.trivial_repr, SO3.irrep(1)]

        for k in range(2, K+1):
            polinomials.append(
                polinomials[-1].tensor(SO3.irrep(1))
            )

        return directsum(polinomials, name=f'polynomial_{K}')

    def forward(self, input: torch.Tensor):
        t = torch.unsqueeze(input[:,0], dim =1)
    
        t2 = GeometricTensor(t, self.in_type)

        encoutsdict2 = {}
        
        output_tensor = []
        for m in range(len(self.blocks)):
            encoutsdict2[f'encout{m}'] = []
        for i in range(t2.shape[-1]):
            t2x = t2[:,:,:,:,i]
            t2x = GeometricTensor(t2x, self.in_type)
            for l, level in enumerate(self.blocks):
                t2x = level(t2x)
                encoutsdict2[f'encout{l}'].append(t2x)
        latentt2 = torch.cat([torch.unsqueeze(t.tensor, dim = 4) for t in encoutsdict2[f'encout{len(self.blocks)-1}']], dim = 4)
        if input.shape[2] == 2:
            a = torch.unsqueeze(input[:,1], dim =1)
            adc = GeometricTensor(a, self.in_type)
            encoutsdicadc = {}
            for m in range(len(self.blocks)):
               encoutsdicadc[f'encout{m}'] = []
            for i in range(adc.shape[-1]):
               adcx = adc[:,:,:,:,i]
               adcx = GeometricTensor(adcx, self.in_type)
               for l, level in enumerate(self.blocks):
                   adcx = level(adcx)
                   encoutsdicadc[f'encout{l}'].append(adcx)
            latentadc = torch.cat([torch.unsqueeze(t.tensor, dim = 4) for t in encoutsdicadc[f'encout{len(self.blocks)-1}']], dim = 4)
        else:
            latentadc = None
        
        latentt2p = GeometricTensor(latentt2, self.latent_type3d)
        latentt2p = self.blockslatent1(latentt2)
        if self.VQ == True:
           vqts, q_lossts, infots = self.quantise(latentt2p)           
        else:
           vqts, q_lossts, infots = out_tp, 0, 0
        latent = self.blockslatent2(vqts)
        for i in range(input.shape[-1]):
            dec = torch.squeeze(latent[:,:,:,:,i].tensor)
            dec = GeometricTensor(dec, self.latent_type2d)
            for j, level in enumerate(self.blocks2):      
                if (j % 2 == 0):
                   dec = level(dec) if j < (len(self.blocks2)-2) else dec
                else:
                   if (len(self.blocks)-((j+1)//2)+1) > 1:
                       dec = tensor_directsum([dec, encoutsdic[f'encout{len(self.blocks)-((j+1)//2+1)}'][i]]) 
                   dec = level(dec)
            output_tensor.append(dec)
    
        out = torch.cat([torch.unsqueeze(t.tensor, dim = 4) for t in output_tensor], dim = 4)
        
        
        return out, latentt2, latentadc, vqts, q_lossts
