import torch
import torch.nn as nn

from src.model.layers import *


class TCN(nn.Module):
    def __init__(
        self,
        channels_in,
        filters=16,
        kernel_size=5,
        dilations=[2**k for k in range(10)],
        dropout_rate=0.1,
    ):
        super(TCN, self).__init__()
        channels_ls = [channels_in]
        channels_ls.extend([filters for i in range(len(dilations) - 1)])
        self.tcn = nn.ModuleList(
            [
                Residual(
                    channels_ls[i], filters, kernel_size, dilations[i], dropout_rate
                )
                for i in range(len(dilations))
            ]
        )

    def forward(self, x):
        skip = 0.0
        for layer in self.tcn:
            x, x_no_res = layer(x)
            skip = skip + x_no_res
        return x, skip


class Head(nn.Module):
    def __init__(self, channels_in, filters=16, dropout_rate=0.1):
        super(Head, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(channels_in, filters)

    def forward(self, x):
        x = self.pool(x)
        x = torch.squeeze(x, -1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class Hat(
    nn.Module
):  # this is called Hat because it is used on top of the Head, get it?
    def __init__(self, channels_in, output_dim=300):
        super(Hat, self).__init__()
        self.classification = nn.Linear(channels_in, output_dim)
        self.regression = nn.Linear(channels_in, 1)

    def forward(self, x):
        return self.classification(x), self.regression(x)


class Siamese(nn.Module):
    def __init__(
        self,
        filters=16,
        dilations=[2**k for k in range(10)],
        dropout_rate=0.1,
        output_dim=300,
    ):
        super(Siamese, self).__init__()
        self.conv1 = nn.Conv2d(1, filters, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=(8, 1))
        self.pool = nn.MaxPool2d((3, 1))
        self.batch_norm = nn.BatchNorm2d(1)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)

        self.tcn = TCN(filters, filters, 5, dilations, dropout_rate)
        self.head = Head(filters, filters, dropout_rate)
        self.hat = Hat(filters, output_dim)

    def forward(self, x):
        x = x.unsqueeze(-3)
        x = self.batch_norm(x)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = torch.squeeze(x, -2)
        x, skip = self.tcn(x)
        x = self.head(
            x
        )  # !!!!!! IMPORTANT !!!!!!: the input to the head should be the aggregate of features from all residuals 'skip', but official implementation simply uses the output of the last residual 'x'
        classification_output, regression_output = self.hat(x)

        return classification_output, regression_output


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
