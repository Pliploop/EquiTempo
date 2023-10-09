import torch
import torch.nn as nn

from src.model.layers import *
    


class Siamese(nn.Module):
    def __init__(self, base_channels=16, mel_dim=128, hop=256, fac=4, num_features=128):
        super(Siamese, self).__init__()
        self.base_channels = base_channels
        self.dim = ((fac*hop)//2)+1
        self.mel_dim = mel_dim

        self.c0 = nn.Conv2d(1, base_channels, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.res1 = ResBlock(base_channels, base_channels*2, kernel_size=(3,3), strides=(2,2), noise=False, padding=(1,1), name='1')
        self.res2 = ResBlock(base_channels*2, base_channels*4, kernel_size=(3,3), strides=(2,2), noise=False, padding=(1,1), name='2')
        self.res3 = ResBlock(base_channels*4, base_channels*8, kernel_size=(3,3), strides=(2,2), noise=False, padding=(1,1), name='3')
        self.res4 = ResBlock(base_channels*8, base_channels*16, kernel_size=(3,3), strides=(2,2), noise=False, padding=(1,1), name='4')
        self.res5 = ResBlock(base_channels*16, base_channels*32, kernel_size=(3,3), strides=(2,2), noise=False, padding=(1,1), name='5')
        self.res6 = ResBlock(base_channels*32, base_channels*32, kernel_size=(3,3), strides=(1,1), noise=False, padding=(1,1), name='6')

        self.proj = nn.Linear(in_features=base_channels*4*4*32, out_features=num_features)

             
    def forward(self, x):
        x = x.unsqueeze(-3)
        x = self.c0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.proj(torch.flatten(x, start_dim=-3))

        return x


# class Siamese(nn.Module):
#     def __init__(self, base_channels=16, mel_dim=128, hop=256, fac=4, num_features=128):
#         super(Siamese, self).__init__()
#         self.base_channels = base_channels
#         self.dim = ((fac*hop)//2)+1
#         self.mel_dim = mel_dim

#         self.c0 = nn.Conv2d(1, base_channels, kernel_size=(3,3), stride=(2,2), padding=(1,1))
#         self.res1 = ResBlock(base_channels, base_channels*2, kernel_size=(3,3), strides=(2,2), noise=False, padding=(1,1), name='1')
#         self.res2 = ResBlock(base_channels*2, base_channels*4, kernel_size=(3,3), strides=(2,2), noise=False, padding=(1,1), name='2')
#         self.res3 = ResBlock(base_channels*4, base_channels*8, kernel_size=(3,3), strides=(2,2), noise=False, padding=(1,1), name='3')
#         self.res4 = ResBlock(base_channels*8, base_channels*16, kernel_size=(3,3), strides=(2,2), noise=False, padding=(1,1), name='4')
#         self.res5 = ResBlock(base_channels*16, base_channels*32, kernel_size=(3,3), strides=(2,2), noise=False, padding=(1,1), name='5')
#         self.res6 = ResBlock(base_channels*32, base_channels*32, kernel_size=(3,3), strides=(1,1), noise=False, padding=(1,1), name='6')

#         self.proj = nn.Linear(in_features=base_channels*4*4*32, out_features=num_features)

             
#     def forward(self, x):
#         x = x.unsqueeze(-3)
#         x = self.c0(x)
#         x = self.res1(x)
#         x = self.res2(x)
#         x = self.res3(x)
#         x = self.res4(x)
#         x = self.res5(x)
#         x = self.res6(x)
#         x = self.proj(torch.flatten(x, start_dim=-3))

#         return x