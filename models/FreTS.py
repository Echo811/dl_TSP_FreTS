import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        '''
            一些常量
        '''
        # 序列嵌入维度
        self.embed_size = 128 #embed_size
        # 隐藏层维度
        self.hidden_size = 256 #hidden_size
        # 预测长度          ε
        self.pre_length = configs.pred_len
        # channels        N
        self.feature_size = configs.enc_in #channels
        # 序列长度
        self.seq_length = configs.seq_len

        '''
            其他常量
        '''
        self.channel_independence = configs.channel_independence
        # 对复数张量 y 进行软阈值化操作中的阈值，也是为了防止计算爆炸
        self.sparsity_threshold = 0.01
        # 定义一个尺度，减少梯度爆炸的可能
        self.scale = 0.02

        '''
            定义张量
        '''
        # 嵌入矩阵
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        # 分别表示【FreCL】中的实部w、b和虚部w、b
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        # 分别表示【FreTL】中的实部w、b和虚部w、b
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        '''
            FFN
        '''
        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    # dimension extension
    '''
        [N*T*1 x 1*D]  ---extension--->  [N*T*D]
    '''
    def tokenEmb(self, x):
        # x: [Batch, Input_length, Channel]
        x = x.permute(0, 2, 1)  # 维度置换
        # 此时x：[Batch, Channel, Input_length]
        x = x.unsqueeze(3)  # 第四个维度上插入一个维度 此时：[Batch, Channel, Input_length, 1]
        y = self.embeddings
        return x * y  # N*T*1 x 1*D = N*T*D

    '''
        频域时间学习器
    '''
    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        # 使用 PyTorch 中的 FFT（快速傅里叶变换）函数对输入张量 x 在第三个维度（时间维度）上进行实部的傅里叶变换。norm='ortho' 表示进行正交归一化。
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
        # 调用频域 MLP 操作的函数
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        # 逆 FFT 函数，将经过频域 MLP 处理后的张量 y 在第三个维度上【也就是T所在的维度】进行逆傅里叶变换
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    '''
        频域通道学习器
    '''
    # frequency channel learner
    def MLP_channel(self, x, B, N, L): # 过程与上面同理
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        '''
            两个张量的维度是 [B, nd, dimension // 2 + 1, self.embed_size]，
            其中 B 是 Batch 大小，nd 是序列数量，dimension 是频域的维度，self.embed_size 是嵌入的维度。
        '''
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)


        # 使用 einsum 函数执行了矩阵乘法操作，其中 x.real 表示输入张量 x 的实部，r 表示实部的权重，
        # rb 表示实部的偏置。然后通过 ReLU 激活函数进行非线性变换。
        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )
        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        # 这一行代码通过 torch.stack 函数将实部和虚部在 [最后一个维度上] 进行堆叠，得到一个复数张量 y
        # 模型架构中的 concat 操作
        y = torch.stack([o1_real, o1_imag], dim=-1)
        # 对复数张量 y 进行软阈值化操作
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        # 将处理后的张量重新转换为复数张量
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        # 【1】 嵌入以及维度扩展
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)
        bias = x

        # 【2】 频域通道学习器
        # [B, N, T, D]
        if self.channel_independence == '1':
            x = self.MLP_channel(x, B, N, T)

        # 【3】 频域时间学习器
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)

        # 相当于残差连接
        x = x + bias

        # 【4】 FFN
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x
