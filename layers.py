import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import einops
import numpy as np
from torch import einsum
from typing import Tuple, Union
from collections import defaultdict

from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *

from scipy import stats


import math
from scipy.cluster.vq import kmeans2
def nonlinearity(x, act='swish'):
    #swish actication if act is swish else pick activation of your choice
    if act == 'swish':
       nl = x*torch.sigmoid(x)
    else:
       act = eval(act) 
       nl = act(x)

    return nl


def Normalise(in_channels, dim, num_groups=None):
    #batch normalisation if num_groups is none otherwise group normalisation
    if num_groups ==None:
        return torch.nn.BatchNorm3d(in_channels) if dim =='3D' else torch.nn.BatchNorm2d(in_channels)
    else:
        assert in_channels % num_groups == 0
        return torch.nn.InstanceNorm3d(in_channels) if dim =='3D' else torch.nn.InstanceNorm2d(in_channels) #torch.nn.GroupNorm(num_groups, in_channels)
        #return torch.nn.GroupNorm(num_groups, in_channels)
class Convblock(nn.Module):
    def __init__(self, *, in_channels,out_channels,dim, kernel, act= None,
                 dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        self.conv = torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernel,stride=1, padding=1) if dim == '2D' else torch.nn.Conv3d(in_channels,out_channels,kernel_size=kernel,stride=1, padding=1)
        self.norm = Normalise(in_channels =  out_channels,  dim = dim)
        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, x):
        h = self.conv(x)
        h = self.norm(h)
        h = nonlinearity(h, act = self.act)
        h = self.dropout(h)


        return h

class Strided_Convblock(nn.Module):
    def __init__(self, *, in_channels,out_channels,dim,kernel,act= None,
                 dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        self.conv = torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernel,stride=2, padding=1) if dim == '2D' else torch.nn.Conv3d(in_channels,out_channels,kernel_size=kernel,stride=2, padding=1)
        self.norm = Normalise(in_channels =  out_channels,  dim = dim)
        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, x):
        h = self.conv(x)
        h = self.norm(h)
        h = nonlinearity(h, act = self.act)
        h = self.dropout(h)


        return h
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels,out_channels,groups,dim,act= None,
                 dropout=0.0):
        super().__init__()
        # Pre-activation convolutional residual block for 2D or 3D input; "dim"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        self.norm1 = Normalise(in_channels = in_channels, num_groups = groups, dim = dim)
        self.conv1 = torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1) if dim == '2D' else torch.nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.norm2 = Normalise(in_channels =  out_channels, num_groups = groups, dim = dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1) if dim == '2D' else torch.nn.Conv3d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        
        if self.in_channels != self.out_channels:
           self.conv_skip = torch.nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0) if dim == '2D' else torch.nn.Conv3d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        h = self.norm1(x)
        h = nonlinearity(h, act=self.act)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h, act = self.act)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
           x = self.conv_skip(x)

        return x+h
class ResnetBlockH(nn.Module):
    def __init__(self, *, in_channels,out_channels,groups,dim,act= None,
                 dropout=0.0):
        super().__init__()
        # Pre-activation convolutional residual block for 2D or 3D input; "dim"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        self.norm1 = Normalise(in_channels = in_channels, num_groups = groups, dim = dim)
        self.conv1 = torch.nn.Conv3d(in_channels,out_channels,kernel_size=(3,3,1),stride=1,padding=(1,1,0)) 
        self.norm2 = Normalise(in_channels =  out_channels, num_groups = groups, dim = dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels,out_channels,kernel_size=(3,3,1),stride=1,padding=(1,1,0))
        if self.in_channels != self.out_channels:
           self.conv_skip = torch.nn.Conv3d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        h = self.norm1(x)
        h = nonlinearity(h, act=self.act)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h, act = self.act)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
           x = self.conv_skip(x)

        return x+h
class NonLocalBlock2D(nn.Module):
    def __init__(self, channels, groups):
        super(NonLocalBlock2D, self).__init__()


        self.gn = Normalise(in_channels =  channels, num_groups = groups, dim = '2D')
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A

class NonLocalBlock3D(nn.Module):
    def __init__(self, channels, groups):
        super(NonLocalBlock3D, self).__init__()


        self.gn = Normalise(in_channels =  channels, num_groups = groups, dim = '3D')
        self.q = nn.Conv3d(channels, channels, 1, 1, 0)
        self.k = nn.Conv3d(channels, channels, 1, 1, 0)
        self.v = nn.Conv3d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv3d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w, z = q.shape

        q = q.reshape(b, c, h*w*z)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w*z)
        v = v.reshape(b, c, h*w*z)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w, z)

        return x + A
class Upsample(nn.Module):
    def __init__(self, in_channels, dim, with_conv):
        super().__init__()
        # Upsampling block for 2D or 3D input; "dim" using linear interpolation with or without a convolutional layer; "with_conv"
        self.with_conv = with_conv
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if dim == '2D' else nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1) if dim == '2D' else torch.nn.Conv3d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)


    def forward(self, x):
        x = self.up(x)
        if self.with_conv:
            x = self.conv(x)
        return x 


class UpsampleH(nn.Module):
    def __init__(self, in_channels, dim, with_conv):
        super().__init__()
        # Upsampling block for 2D or 3D input; "dim" using linear interpolation with or without a convolutional layer; "with_conv"
        self.with_conv = with_conv
        self.up = nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=False)
        if self.with_conv:
            self.conv =  torch.nn.Conv3d(in_channels,in_channels,kernel_size=(3,3,1),stride=1,padding=(1,1,0))


    def forward(self, x):
        x = self.up(x)
        if self.with_conv:
            x = self.conv(x)
        return x
class Encoder(nn.Module):
    """
      This class creates an encoder with any number of "levels" with each level
      with each level consisting of any number of pre activation residual convolutional  "blocks"
      for handling 2D or 3D dimension ("dim") input
    """

    def __init__(self, in_ch, channels,groups, blocks, dim, act= None,  dropout=0.0):
        super().__init__()
        self.enc = nn.ModuleList()
        levels = len(blocks)
        for i in range(levels):
            block = []
            for j in range(blocks[i]):
                out_channels = channels *2 if j == 0 else channels                        
                if i == 0:
                   block.append(torch.nn.Conv2d(in_ch, channels, kernel_size = 3, stride = 1, padding = 1) if dim == '2D' else torch.nn.Conv3d(in_ch, channels, kernel_size = 3, stride = 1, padding = 1))
                block.append(ResnetBlock(in_channels = channels, out_channels = out_channels,dim = dim,  groups = groups, act = act, dropout = dropout))          
                channels = out_channels
            self.enc.append(nn.Sequential(*block))
            if i != levels:
               self.enc.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2) if dim == '2D' else nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=2))
            #self.enc.append(NonLocalBlock2D(channels=out_channels, groups=groups) if dim == '2D' else NonLocalBlock3D(channels=out_channels, groups=groups))
    def forward(self, x):
    
        for l, level in enumerate(self.enc):
            x = level(x)
        return x
class CrossEncoder(nn.Module):
    """
      This class creates an encoder with any number of "levels" with each level
      with each level consisting of any number of pre activation residual convolutional  "blocks"
      for handling 2D or 3D dimension ("dim") input
      """

    def __init__(self, in_ch, channels,groups, blocks, dim, act= None,  dropout=0.0):
        super().__init__()
        self.enc = nn.ModuleList()
        levels = len(blocks)
        for i in range(levels):
            block = []
            for j in range(blocks[i]):
                out_channels = channels *2 if j == 0 else channels                        
                if i == 0:
                   block.append(torch.nn.Conv2d(in_ch, channels, kernel_size = 3, stride = 1, padding = 1) if dim == '2D' else torch.nn.Conv3d(in_ch, channels, kernel_size = 3, stride = 1, padding = 1))
                block.append(ResnetBlock(in_channels = channels, out_channels = out_channels,dim = dim,  groups = groups, act = act, dropout = dropout))          
                channels = out_channels
            self.enc.append(nn.Sequential(*block))
            if i != levels:
               self.enc.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2) if dim == '2D' else nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=2))

    def forward(self, x):
        outs = []
        for l, level in enumerate(self.enc):
            x = level(x)
            if l % 2 ==0:
               outs.append(x)
        return outs
class CrossEncoderH(nn.Module):
    """
      This class creates an encoder with any number of "levels" with each level
      with each level consisting of any number of pre activation residual convolutional  "blocks"
      for handling 2D or 3D dimension ("dim") input
      """

    def __init__(self, in_ch, channels,groups, blocks,TD, dim, act= None,  dropout=0.0):
        super().__init__()
        self.enc = nn.ModuleList()
        levels = len(blocks)-TD
        for i in range(levels):
            block = []
            for j in range(blocks[i]):
                out_channels = channels *2 if j == 0 else channels
                if i == 0:
                   block.append(torch.nn.Conv2d(in_ch, channels, kernel_size = 3, stride = 1, padding = 1) if dim == '2D' else torch.nn.Conv3d(in_ch, channels, kernel_size = (3,3,1), stride = 1, padding = (1,1,0)))
                block.append(ResnetBlockH(in_channels = channels, out_channels = out_channels,dim = dim,  groups = groups, act = act, dropout = dropout))
                channels = out_channels
            self.enc.append(nn.Sequential(*block))
            if i !=levels:
               self.enc.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2) if dim == '2D' else nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,1), padding=(1,1,0), stride=(2,2,1)))
        for i in range(TD):
            block = []
            for j in range(blocks[levels-1+i]):
                out_channels = channels *2 if j == 0 else channels
                block.append(ResnetBlock(in_channels = channels, out_channels = out_channels,dim = dim,  groups = groups, act = act, dropout = dropout))
                channels = out_channels
            self.enc.append(nn.Sequential(*block))
            if (levels-1+i) != len(blocks):
               self.enc.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2) if dim == '2D' else nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)) 
    def forward(self, x):
        outs = []
        for l, level in enumerate(self.enc):
            x = level(x)
            if l % 2 ==0:
               outs.append(x)
        return outs

class EncoderH(nn.Module):
    """
      This class creates an encoder with any number of "levels" with each level
      with each level consisting of any number of pre activation residual convolutional  "blocks"
      for handling 2D or 3D dimension ("dim") input
      """

    def __init__(self, in_ch, channels,groups, blocks,TD, dim, act= None,  dropout=0.0):
        super().__init__()
        self.enc = nn.ModuleList()
        levels = len(blocks)-TD
        for i in range(levels):
            block = []
            for j in range(blocks[i]):
                out_channels = channels *2 if j == 0 else channels
                if i == 0:
                   block.append(torch.nn.Conv2d(in_ch, channels, kernel_size = 3, stride = 1, groups = groups, padding = 1) if dim == '2D' else torch.nn.Conv3d(in_ch, channels, kernel_size = (3,3,1), stride = 1, groups = groups,padding = (1,1,0)))
                block.append(ResnetBlockH(in_channels = channels, out_channels = out_channels,dim = dim,  groups = groups, act = act, dropout = dropout))
                channels = out_channels
            self.enc.append(nn.Sequential(*block))
            if i != levels:
               self.enc.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2) if dim == '2D' else nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,1), padding=(1,1,0), stride=(2,2,1)))
        for i in range(TD):
            block = []
            for j in range(blocks[levels-1+i]):
                out_channels = channels *2 if j == 0 else channels
                block.append(ResnetBlock(in_channels = channels, out_channels = out_channels,dim = dim,  groups = groups, act = act, dropout = dropout))
                channels = out_channels
            self.enc.append(nn.Sequential(*block))
            if (levels-1+i) != len(blocks):
               self.enc.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2) if dim == '2D' else nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=2))
    def forward(self, x):
        for l, level in enumerate(self.enc):
            x = level(x)
        return x
class Decoder(nn.Module):
      """
      This class creates a decoder with any number of "levels" with each level
      with each level consisting of any number of pre activation residual convolutional  "blocks"
      for handling 2D or 3D dimension ("dim") input
      """
      def __init__(self, channels,out_ch, groups,  blocks, dim, act= None,  with_conv = True, dropout=0.0):
        super().__init__()
        levels = len(blocks)
        self.dec = nn.ModuleList()
        for i in range(levels):
            block = []
            self.dec.append(Upsample(in_channels = channels, dim=dim,
                with_conv = with_conv))
            for j in range(blocks[i]):
                if j == 0:
                   out_channels = (channels//2)
                else:
                   out_channels = channels
                block.append(ResnetBlock(in_channels = channels, out_channels = out_channels, dim = dim, groups = groups, act = act, dropout = dropout))
                channels = out_channels
                if i == levels-1:
                   block.append(torch.nn.Conv2d(channels, out_ch, kernel_size = 3, stride = 1, padding = 1) if dim == '2D' else torch.nn.Conv3d(channels, out_ch, kernel_size = 3, stride = 1, padding = 1))
            self.dec.append(nn.Sequential(*block))
      def forward(self, x):
        for j, level in enumerate(self.dec):
            x = level(x)
        return x

class CrossDecoder(nn.Module):
      """
      This class creates a decoder with any number of "levels" with each level
      with each level consisting of any number of pre activation residual convolutional  "blocks" 
      for handling 2D or 3D dimension ("dim") input
      """
      def __init__(self, channels,groups,out_ch, blocks, dim, act= None,  with_conv = True, dropout=0.0):
        super().__init__()
        levels = len(blocks) 
        self.dec = nn.ModuleList()
        #self.dec.append(NonLocalBlock2D(channels=channels, groups=groups) if dim == '2D' else NonLocalBlock3D(channels=channels, groups=groups))
        for i in range(levels):
            block = []
            self.dec.append(Upsample(in_channels = channels, dim=dim,  with_conv = with_conv))
            for j in range(blocks[i]):
                channels = channels + channels//2
                if j == 0:
                   out_channels = (channels//3)
                else: 
                   out_channels = channels
                block.append(ResnetBlock(in_channels = channels, out_channels = out_channels, dim = dim, groups = groups, act = act, dropout = dropout))
                channels = out_channels
                if i == levels-1:
                   block.append(torch.nn.Conv2d(channels, out_ch, kernel_size = 3, stride = 1, padding = 1) if dim == '2D' else torch.nn.Conv3d(channels, out_ch, kernel_size = 3, stride = 1, padding = 1)) 
            self.dec.append(nn.Sequential(*block))
      def forward(self, x):
        x1 = x[-1]
        enc = list(reversed(x))
        for j, level in enumerate(self.dec):
            if j % 2 == 0:
               x1 = level(x1)
            else:
               x1 = level(torch.cat((enc[(j+1)//2], x1), dim = 1))
        return x1

class CrossDecoderH(nn.Module):
      """
      This class creates a decoder with any number of "levels" with each level
      with each level consisting of any number of pre activation residual convolutional  "blocks"
      for handling 2D or 3D dimension ("dim") input
      """
      def __init__(self, channels,groups,out_ch, blocks, TD, dim, act= None,  with_conv = True, dropout=0.0):
        super().__init__()
        levels = len(blocks)-TD
        self.dec = nn.ModuleList()
        #self.dec.append(NonLocalBlock2D(channels=channels, groups=groups) if dim == '2D' else NonLocalBlock3D(channels=channels, groups=groups))
        for i in range(TD):
            block = []
            if i < TD-1:
               self.dec.append(Upsample(in_channels = channels, dim=dim,  with_conv = with_conv))
            else:
               self.dec.append(UpsampleH(in_channels = channels, dim=dim,  with_conv = with_conv)) 
            for j in range(blocks[i]):
                channels = channels + channels//2
                if j == 0:
                    out_channels = (channels//3)
                else:
                    out_channels = channels
                if TD > 1:
                    block.append(ResnetBlock(in_channels = channels, out_channels = out_channels, dim = dim, groups = groups, act = act, dropout = dropout))
                else:
                    block.append(ResnetBlockH(in_channels = channels, out_channels = out_channels,dim = dim,  groups = groups, act = act, dropout = dropout))
                channels = out_channels
                self.dec.append(nn.Sequential(*block))
        for i in range(levels):
            block = []
            self.dec.append(UpsampleH(in_channels = channels, dim=dim,  with_conv = with_conv))
            for j in range(blocks[i+TD]):
                channels = channels + channels//2
                if j == 0:
                   out_channels = (channels//3)
                else:
                   out_channels = channels
                block.append(ResnetBlock(in_channels = channels, out_channels = out_channels, dim = dim, groups = groups, act = act, dropout = dropout))
                channels = out_channels
                if i == levels-1:
                   block.append(torch.nn.Conv2d(channels, out_ch, kernel_size = 3, stride = 1, padding = 1) if dim == '2D' else torch.nn.Conv3d(channels, out_ch, kernel_size = (3,3,1), stride = 1, padding = (1,1,0)))
            self.dec.append(nn.Sequential(*block))
      def forward(self, x):  
        x1 = x[-1]
        enc = list(reversed(x))
        for j, level in enumerate(self.dec):
            if j % 2 == 0:
               x1 = level(x1)
            else:
               x1 = level(torch.cat((enc[(j+1)//2], x1), dim = 1))

        return x1

class DecoderH(nn.Module):
      """
      This class creates a decoder with any number of "levels" with each level
      with each level consisting of any number of pre activation residual convolutional  "blocks"
      for handling 2D or 3D dimension ("dim") input
      """
      def __init__(self, channels,groups,out_ch, blocks, TD, dim, act= None,  with_conv = True, dropout=0.0):
        super().__init__()
        levels = len(blocks)-TD
        self.dec = nn.ModuleList()
        #self.dec.append(NonLocalBlock2D(channels=channels, groups=groups) if dim == '2D' else NonLocalBlock3D(channels=channels, groups=groups))
        for i in range(TD):
            block = []
            if i < TD-1:
               self.dec.append(Upsample(in_channels = channels, dim=dim,  with_conv = with_conv))
            else:
               self.dec.append(UpsampleH(in_channels = channels, dim=dim,  with_conv = with_conv))
            for j in range(blocks[i]):
                #channels = channels #+ channels//2
                if j == 0:
                    out_channels = channels//2
                else:
                    out_channels = channels
                if TD > 1:
                    block.append(ResnetBlock(in_channels = channels, out_channels = out_channels, dim = dim, groups = groups, act = act, dropout = dropout))
                else:
                    block.append(ResnetBlockH(in_channels = channels, out_channels = out_channels,dim = dim,  groups = groups, act = act, dropout = dropout))
                channels = out_channels
                self.dec.append(nn.Sequential(*block))
        for i in range(levels):
            block = []
            self.dec.append(UpsampleH(in_channels = channels, dim=dim,  with_conv = with_conv))
            for j in range(blocks[i+TD]):
                #channels = channels #+ channels//2
                if j == 0:
                   out_channels = channels//2
                else:
                   out_channels = channels
                block.append(ResnetBlockH(in_channels = channels, out_channels = out_channels, dim = dim, groups = groups, act = act, dropout = dropout))
                channels = out_channels
                if i == levels-1:
                   block.append(torch.nn.Conv2d(channels, out_ch, kernel_size = 3, stride = 1, groups = groups, padding = 1) if dim == '2D' else torch.nn.Conv3d(channels, out_ch, kernel_size = (3,3,1), stride = 1, groups = groups,padding = (1,1,0)))
            self.dec.append(nn.Sequential(*block))
      def forward(self, x1):
        dec1 = []
        #x1 = x[-1]
        #enc = list(reversed(x))
        for j, level in enumerate(self.dec):
            if j % 2 == 0:
               x1 = level(x1)
            else:
               x1 = level(x1)
               dec1.append(x1)
        return dec1

class VectorQuantiser(nn.Module):
    """
    This class is adapted from https://github.com/CompVis/taming-transformers
    https://arxiv.org/pdf/2012.09841.pdf
    """
    def __init__(self, n_e, e_dim, beta = 0.2, dim = '3D', quantise = 'spatial', legacy=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.dim = dim
        self.legacy = legacy
        self.quantise = quantise
        

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.register_buffer('data_initialized', torch.zeros(1))
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.ed = lambda x: [torch.norm(x[i]) for i in range(x.shape[0])]

    def dist(self, u, v):
        d = torch.sum(u ** 2, dim=1, keepdim=True) + \
             torch.sum(v**2, dim=1) - 2 * \
             torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))

        return d
    
    def geodist(self, u, v):
        d1 = torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
        ed1 = torch.sqrt(torch.sum(u ** 2, dim=1, keepdim=True))
        ed2 = torch.sqrt(torch.sum(v**2, dim=1, keepdim = True))
        ed3 = torch.einsum('bd,dn->bn', ed1, rearrange(ed2, 'n d -> d n'))
        geod = torch.clamp(d1/(ed3), min=-0.99999, max=0.99999)
        
        return torch.acos(geod)
 
    def forward(self, z):
        #Determine to quantise either spatial or channel wise
        if self.quantise == 'spatial':
           # reshape z  and flatten
           if self.dim == '2D':
               z = rearrange(z, 'b c h w -> b h w c').contiguous()
           else:    
               z = rearrange(z, 'b c h w z -> b h w z c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        flatten = z.reshape(-1, self.e_dim)
        if  self.data_initialized.item() == 0:
            print('running kmeans!!') # data driven initialization for the embeddings
            rp = torch.randperm(flatten.size(0))
            kd = kmeans2(flatten[rp[:20000]].data.cpu().numpy(), self.n_e, minit='points')
            self.embedding.weight.data.copy_(torch.from_numpy(kd[0]))
            self.data_initialized.fill_(1)
        
        # compute distances from z to codebook
        d = self.dist(z_flattened, self.embedding.weight)
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # compute mean codebook distances
        cd = self.dist(self.embedding.weight, self.embedding.weight)
        min_distance = torch.kthvalue(cd, 2, 0)
        mean_cb_distance = torch.mean(min_distance[0])
            
        # compute mean codebook variance
        mean_cb_variance = torch.mean(torch.var(cd, 1))

        # compute loss for embedding
        
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)
    
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape to original input shape
        if self.quantise == 'spatial':
           if self.dim == '2D':
              z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
           else:
              z_q = rearrange(z_q, 'b h w z c -> b c h w z').contiguous()
        
        # Get Sampled Indices
        sampled_idx = torch.zeros(z_flattened.shape[0]*self.n_e).to(z.device)
        sampled_idx[min_encoding_indices] = 1
        sampled_idx = sampled_idx.view(z_flattened.shape[0], self.n_e)
        return z_q, loss, (min_encoding_indices, sampled_idx, mean_cb_distance, mean_cb_variance)
    
class EqResBlock3D(EquivariantModule):

    def __init__(self, in_type: FieldType, channels: int, out_type: FieldType = None, stride: int = 1, features: str = '2_96'):

        super(EqResBlock2D, self).__init__()

        self.in_type = in_type
        if out_type is None:
            self.out_type = self.in_type
        else:
            self.out_type = out_type

        self.gspace = self.in_type.gspace

        if args.features == 'ico':
            L = 2
            grid = {'type': 'ico'}
        elif args.features == '2_96':
            L = 2
            grid = {'type': 'thomson_cube', 'N': 4}
        elif argfeatures == '2_72':
            L = 2
            grid = {'type': 'thomson_cube', 'N': 3}
        elif features == '3_144':
            L = 3
            grid = {'type': 'thomson_cube', 'N': 6}
        elif features == '3_192':
            L = 3
            grid = {'type': 'thomson_cube', 'N': 8}
        elif features == '3_160':
            L = 3
            grid = {'type': 'thomson', 'N': 160}
        else:
            raise ValueError()

        so3: SO3 = self.in_type.fibergroup

        # number of samples for the discrete Fourier Transform
        S = len(so3.grid(**grid))

        # We try to keep the width of the model approximately constant
        _channels = channels / S
        _channels = int(round(_channels))

        # Build the non-linear layer
        # Internally, this module performs an Inverse FT sampling the `_channels` continuous input features on the `S`
        # samples, apply ELU pointwise and, finally, recover `_channels` output features with discrete FT.
        ftelu = FourierELU(self.gspace, _channels, irreps=so3.bl_irreps(L), inplace=True, **grid)
        res_type = ftelu.in_type

        print(f'ResBlock: {in_type.size} -> {res_type.size} -> {self.out_type.size} | {S*_channels}')

        self.res_block = SequentialModule(
            R3Conv(in_type, res_type, kernel_size=3, padding=1, bias=False, initialize=False),
            IIDBatchNorm3d(res_type, affine=True),
            ftelu,
            R3Conv(res_type, self.out_type, kernel_size=3, padding=1, stride=stride, bias=False, initialize=False),
        )

        if stride > 1:
            self.downsample = PointwiseAvgPoolAntialiased3D(in_type, .33, 2, 1)
        else:
            self.downsample = lambda x: x

        if self.in_type != self.out_type:
            self.skip = R3Conv(self.in_type, self.out_type, kernel_size=1, padding=0, bias=False)
        else:
            self.skip = lambda x: x

    def forward(self, input: GeometricTensor):

        assert input.type == self.in_type
        return self.skip(self.downsample(input)) + self.res_block(input)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]
        else:
            return input_shape

class EqResBlock2D(EquivariantModule):

    def __init__(self, in_type: FieldType, out_type: FieldType, repr: str = 'Regular', stride: int = 1):

        super(EqResBlock2D, self).__init__()

        self.in_type = in_type
        if out_type is None:
            self.out_type = self.in_type
        else:
            self.out_type = out_type
        
        if repr == 'Regular':
           nonlinearity = ReLU(out_type)
        else:
           norm_relu = NormNonLinearity(out_type)
           nonlinearity = MultipleModule(
                    self.out_type,
                    ['norm']*len(vector_field),
                    [(norm_relu, 'norm')]
           )
        self.res_block = SequentialModule(
             R2Conv(in_type, out_type, kernel_size=3, padding = 1, bias=False, initialize=False),
             InnerBatchNorm(out_type),
             nonlinearity,
             R2Conv(out_type, out_type, kernel_size=3, padding =1, bias=False, initialize=False),
             nonlinearity,
        )

        if stride > 1:
            self.downsample = PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        else:
            self.downsample = lambda x: x

        if self.in_type != self.out_type:
            self.skip = R2Conv(self.in_type, self.out_type, kernel_size=1, padding=0, bias=False)
        else:
            self.skip = lambda x: x

    def forward(self, input: GeometricTensor):

        assert input.type == self.in_type
        return self.downsample(self.skip(input) + self.res_block(input))

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]
        else:
            return input_shape

class SE3Encoder(nn.Module):

    def __init__(self,args):

        super(SE3CNN, self).__init__()

        self.gs = rot3dOnR3()

        self.num_features = args.num_features
        self.multiplicity = args.multiplicity
        self.so3channels = args.so3channels

        self.in_type = FieldType(self.gs, [self.gs.trivial_repr])

        self._init = args.init
        layer_types = []
        for i in range(len(self.num_features)):
            layer_types.append((FieldType(self.gs, [self.build_representation(self.num_features[i])] * self.multiplicity[i]), self.so3channels[i]))
        blocks = [
            R3Conv(self.in_type, layer_types[0][0], kernel_size=5, padding=2, bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                ResBlock(layer_types[i][0], layer_types[i][1], layer_types[i+1][0], 2, features=args.res_features)
            )

        # For pooling, we map the features to a spherical representation (bandlimited to freq 2)
        # Then, we apply pointwise ELU over a number of samples on the sphere and, finally, compute the average
        # # (i.e. recover only the frequency 0 component of the output features)
        if args.pool == "icosidodecahedron":
            # samples the 30 points of the icosidodecahedron
            # this is only perfectly equivarint to the 12 tethrahedron symmetries
            grid = self.gs.fibergroup.sphere_grid(type='ico')
        elif args.pool == "snub_cube":
            # samples the 24 points of the snub cube
            # this is perfectly equivariant to all 24 rotational symmetries of the cube
            grid = self.gs.fibergroup.sphere_grid(type='thomson_cube', N=1)
        else:
            raise ValueError(f"Pooling method {pool} not recognized")
    

        self.blocks = SequentialModule(*blocks)


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

        input = GeometricTensor(input, self.in_type)

        features = self.blocks(input)

        return features


