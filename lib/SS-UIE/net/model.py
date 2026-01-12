import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net.blocks import *
dtype = torch.cuda.FloatTensor 






class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))  #tureL: direct modified x, false: new object and the modified
        self.add_module('conv', nn.Conv2d(in_channels, channels, 3, 1, 1))  #bias: defautl: ture on pytorch, learnable bias


class BNReLUMamba(nn.Sequential):
    def __init__(self, in_channels, channels, drop_rate, H, W, inplace=True):
        super(BNReLUMamba, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))  #tureL: direct modified x, false: new object and the modified
        self.add_module('mamba', SF_Block(in_channels, channels, drop_rate, H, W))  #bias: defautl: ture on pytorch, learnable bias

class GateUnit(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(GateUnit, self).__init__()
        self.add_module('bn',nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels,1,1,0))


class ResidualBlock(torch.nn.Module):

    def __init__(self, channels, drop_rate, H, W):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUMamba(channels, channels, drop_rate, H, W, True)
        self.relu_conv2 = BNReLUMamba(channels, channels, drop_rate, H, W, True)
        
    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""
    def __init__(self, channels, num_resblock, num_memblock, drop_rate, H, W):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels, drop_rate, H, W) for i in range(num_resblock)]
        )
        #self.gate_unit = BNReLUConv((num_resblock+num_memblock) * channels, channels, True)  #kernel 3x3
        self.gate_unit = GateUnit((num_resblock+num_memblock) * channels, channels, True)   #kernel 1x1

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)
       
        
        #gate_out = self.gate_unit(torch.cat([xs,ys], dim=1))
        gate_out = self.gate_unit(torch.cat(xs+ys, 1))  #where xs and ys are list, so concat operation is xs+ys
        ys.append(gate_out)
        return gate_out











class SS_UIE_model(nn.Module):
    def __init__(self, in_channels=3, channels=16, num_memblock=6, num_resblock=6, drop_rate=0.0, H=256, W=256):
        super(SS_UIE_model, self).__init__()

        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]  #config是sc_block总层数
        self.extra_conv1 = BNReLUConv(in_channels, channels, True)
        self.extra_conv2 = BNReLUConv(channels, channels*2, True)  #FENet: staic(bn)+relu+conv1
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)
        self.extra_conv3 = BNReLUConv(channels*2, channels*4, True)  #FENet: staic(bn)+relu+conv1

        self.recons_conv1=BNReLUConv(channels*4, channels*2, True) #ReconNet: static(bn)+relu+conv 
        self.recons_conv2=BNReLUConv(channels*2, channels, True) #ReconNet: static(bn)+relu+conv 
        self.recons_conv3=BNReLUConv(channels, in_channels, True) #ReconNet: static(bn)+relu+conv 

        self.fusion = BNReLUConv(channels*4, channels*4, True) 




        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels*4, num_resblock, i+1, drop_rate, H//4, W//4) for i in range(num_memblock)]
        )
        #ModuleList can be indexed like a regular Python list, but modules it contains are 
        #properly registered, and will be visible by all Module methods.
        
        
        self.weights = nn.Parameter((torch.ones(1, num_memblock)/num_memblock), requires_grad=True)  
        #output1,...,outputn corresponding w1,...,w2



    #Multi-supervised MemNet architecture
    def forward(self, x):
        residual0 = x
        out = self.extra_conv1(x)
        residual1 = out
        out = self.extra_conv2(out)
        out = self.pool(out)
        residual2 = out
        out = self.extra_conv3(out)
        out = self.pool(out)
        residual3 = out

        w_sum=self.weights.sum(1)  
        mid_feat=[]   # A lsit contains the output of each memblock
        ys = [out]  #A list contains previous memblock output(long-term memory)  and the output of FENet
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)  #out is the output of GateUnit  channels=64
            mid_feat.append(out);
        #pred = Variable(torch.zeros(x.shape).type(dtype),requires_grad=False)
        pred = (self.fusion(mid_feat[0])+residual3)*self.weights.data[0][0]/w_sum
        for i in range(1,len(mid_feat)):
            pred = pred + (self.fusion(mid_feat[i])+residual3)*self.weights.data[0][i]/w_sum

        pred = F.interpolate(pred, scale_factor=2)
        pred = self.recons_conv1(pred)
        pred=residual2+pred
        pred = F.interpolate(pred, scale_factor=2)
        pred = self.recons_conv2(pred)
        pred=residual1+pred
        pred = self.recons_conv3(pred)
        pred=residual0+pred


        return pred



