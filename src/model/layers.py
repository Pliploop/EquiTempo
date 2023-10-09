import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat



class Residual(nn.Module):
    def __init__(self, channels_in, filters, kernel_size=5, dilation_rate=1, dropout_rate=0.1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv1d(channels_in, filters, kernel_size=kernel_size, dilation=dilation_rate, padding='same')
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=1, padding=0)
        self.res_conv = nn.Conv1d(channels_in, filters, kernel_size=1, padding=0)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return self.res_conv(y) + x, x


class TCN(nn.Module):
    def __init__(self, channels_in, filters=16, kernel_size=5, dilations=[2**k for k in range(10)], dropout_rate=0.1):
        super(TCN, self).__init__()
        channels_ls = [channels_in].extend([filters for i in range(len(dilations)-1)])
        self.tcn = nn.ModuleList([Residual(channels_ls[i], filters, kernel_size, dilations[i], dropout_rate) for i in range(len(dilations))])

    def forward(self,x):
        skip = 0.
        for layer in self.tcn:
            x, x_no_res = layer(x)
            skip = skip+x_no_res
        return x, skip


class Head(nn.Module):
    def __init__(self, channels_in, filters=16, dropout_rate=0.1):
        super(Head, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(channels_in, filters)

    def forward(self,x):
        x = self.pool(x)
        x = torch.squeeze(x, -1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class Hat(nn.Module): # this is called Hat because it is used on top of the Head, get it?
    def __init__(self, channels_in, output_dim=300):
        super(Hat, self).__init__()
        self.classification = nn.Linear(channels_in, output_dim)
        self.regression = nn.Linear(channels_in, 1)

    def forward(self,x):
        return self.regression(x), self.classification(x)


class Model(nn.Module):
    def __init__(self, filters=16, dilations=[2**k for k in range(10)], dropout_rate=0.1, output_dim=300):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, filters, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=(8,1))
        self.pool = nn.MaxPool2d((3,1))
        self.batch_norm = nn.BatchNorm2d(1)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)

        self.tcn = TCN(filters, filters, 5, dilations, dropout_rate)
        self.head = Head(filters, filters, dropout_rate)
        self.hat = Hat(filters, output_dim)

    def forward(self, x):
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

        x = torch.squeeze(x,-2)
        x, skip = self.tcn(x)
        x = self.head(x) # !!!!!! IMPORTANT !!!!!!: the input to the head should be the aggregate of features from all residuals 'skip', but official implementation simply uses the output of the last residual 'x'
        classification_output, regression_output = self.hat(x)

        return classification_output, regression_output






























# adapted from https://github.com/lucidrains/x_transformers/x_transformers.py
class DynamicPositionBias(nn.Module):
    def __init__(self, dim, heads, depth = 1, log_distance = False, norm = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim) if norm else nn.Identity(),
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device = device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias


# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch_1d.py
class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, pos_emb = True):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.pos_emb = pos_emb

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

        if pos_emb:
            self.dpb = DynamicPositionBias(dim//2, heads)

    def forward(self, x, use_dist=False):
        x = x.squeeze(-2)
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        if use_dist:
            sim = -torch.linalg.vector_norm(q.unsqueeze(-1)-k.unsqueeze(-2), ord=2, dim=-3)
        else:
            sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)

        if self.pos_emb:
            sim = sim + self.dpb(q.shape[-1], k.shape[-1])

        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out).unsqueeze(-2)


class AddNoise(nn.Module):
    def __init__(self, channels):
        super(AddNoise, self).__init__()
        self.b = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)

    def forward(self, inputs):
        rand = torch.randn([inputs.shape[0],inputs.shape[1],1,1], dtype=torch.float32)
        output = inputs + self.b.to(inputs.device) * rand.to(inputs.device)
        return output
    
    
def pixel_shuffle(x, factor=2):
    bs_dim, c_dim, h_dim, w_dim = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    x = x.permute(0,2,3,1)
    x = x.reshape(bs_dim, h_dim, w_dim, c_dim//factor, factor)
    x = x.permute(0,1,2,4,3)
    x = x.reshape(bs_dim, h_dim, w_dim*factor, c_dim//factor)
    return x.permute(0,3,1,2)
    
    
class Adain(nn.Module):
    def __init__(self, in_features, out_features):
        super(Adain, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
    
    def forward(self, x, emb):
        emb = self.linear(emb)
        x = x / (torch.std(x, dim=-1, keepdim=True) + 1e-5)
        return x * torch.unsqueeze(torch.unsqueeze(emb, -1), -1)
    


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)






class ResBlock(nn.Module):
    def __init__(self, inp, filters, kernel_size=(1,4), strides=(1,1), upsample=False, emb=False, noise=False, norm=True, padding='same', cond_dim=32, name='0', leaky=False, attention=False):
        super(ResBlock, self).__init__()
        self.norm = norm
        self.upsample = upsample
        self.noise = noise
        self.emb = emb
        self.padding = padding
        self.name = name
        self.strides = strides
        self.noise = noise
        self.leaky = leaky
        self.kernel_size = kernel_size
        self.attention = attention
        self.conv1 = nn.Conv2d(inp, filters, kernel_size=kernel_size, stride=(1,1), padding=padding)
        self.ln = nn.LayerNorm(inp)
        if upsample:
            self.conv2 = nn.ConvTranspose2d(filters, filters, kernel_size=kernel_size, stride=strides, padding=(0,1))
            self.res_conv = nn.Conv2d(inp//2, filters, kernel_size=(1,1), stride=(1,1), padding=0)
        else:
            self.conv2 = nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=strides, padding=padding)
            self.res_conv = nn.Conv2d(inp, filters, kernel_size=(1,1), stride=(1,1), padding=0)
        if self.emb:
            self.adain = Adain(cond_dim, filters)
        if self.noise:
            self.add_noise = AddNoise(filters)
        if self.attention:
            self.ln_att = nn.LayerNorm(filters, eps=0.001)
            self.mha = Attention(filters)
            

    def forward(self, x):
        if self.emb:
            x, cond = x
        y = x.clone()
        if self.norm:
            x = x.permute(0, 2, 3, 1)
            x = self.ln(x)
            x = x.permute(0, 3, 1, 2)
        else:
            if self.leaky:
                x = nn.functional.leaky_relu(x, negative_slope=0.2)
            else:
                x = F.silu(x)
        x = self.conv1(x)
        x = torch.sqrt(torch.tensor(0.5))*x
        if self.noise:
            x = self.add_noise(x)
        if self.emb:
            x = self.adain(x, cond)
        if self.leaky:
            x = nn.functional.leaky_relu(x, negative_slope=0.2)
        else:
            x = F.silu(x)
        x = self.conv2(x)
        x = torch.sqrt(torch.tensor(0.5))*x
        if self.upsample:
            y = pixel_shuffle(y)
        elif self.strides!=(1,1) and (self.padding=='same' or self.padding!=(0,0)):
            y = F.avg_pool2d(y, self.strides, padding=(y.shape[-2]%self.strides[0],y.shape[-1]%self.strides[1]))
        elif self.padding=='valid':
            y = F.avg_pool2d(y, self.kernel_size, self.strides)
        else:
            y = y
        if y.shape[-3]!=x.shape[-3]:
            y = self.res_conv(y)
        x = x+y

        if self.attention:
            y = x.clone()
            if self.norm:
                x = x.permute(0, 2, 3, 1)
                x = self.ln_att(x)
                x = x.permute(0, 3, 1, 2)
            x = self.mha(x)
            x = x+y

        return x