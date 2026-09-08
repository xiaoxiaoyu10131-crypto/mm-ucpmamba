import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
# from utils.gensp.ssn_sp import ssn_iter
from src.backbones.ssndiedai import ssn_iter

class DWConv1(nn.Module):
    def __init__(self, dim=1024):
        super(DWConv1, self).__init__()
        self.dwconv = nn.Conv1d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=1024):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class MHCA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

class MHSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # chunk, split the tensor to a list
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, MHCA(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FFN(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, MHSA(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FFN(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class GlobalAggregation(nn.Module):

    def __init__(self, channel):

        super(GlobalAggregation, self).__init__()

        head = 4

        self.transformer_encoder = Transformer(dim=channel, depth=1, heads=head,
                                               dim_head=channel // head,
                                               mlp_dim=2 * channel, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=channel, depth=1,
                                                      heads=head, dim_head=channel // head, mlp_dim=2 * channel,
                                                      dropout=0,
                                                      softmax=True)

        self.pos = DWConv1(channel)
        # 对区域描述符 des 进行通道维度的特征处理。

        self.channel = channel
    #     保存输入特征的通道数。

    def forward(self, des, x):
        # 区域描述符，shape 为 [batch_size, sp_num, channel]。
        # 每个样本的区域特征描述符
        # x每个特征图的 shape 为 [channel, height, width]。
        c,h,w = x[0].shape
        # 提取每个样本的 channel, height, width，并定义一个空列表 gre 存储重建后的全局特征图：

        batch = len(des)

        gre = []
        # 对每个样本（批次）进行如下处理：
        for i in range(batch):

            di = des[i].unsqueeze(0)
            # 1, s, c
            # [1, sp_num, channel] - 增加 batch 维度

            xpi = self.pos(di.permute(0, 2, 1)).permute(0,2,1)
            # 使用深度可分离卷积 DWConv1 处理区域描述符 des，用于提取局部通道特征
            # self.pos 作用于 di 并处理通道维度。
            # permute 将数据重新排列回 [1, sp_num, channel] 格式。

            xi = x[i].permute(1, 2, 0).reshape(-1, self.channel)
            # 将输入特征图 x 拉平成 [n, channel] 格式，其中 n = height * width：

            # 将处理后的区域描述符与 pos 卷积结果相加，形成编码器输入
            encoder_qkv = di + xpi
            # # [1, sp_num, channel]

            decoder_q = xi.unsqueeze(0)
            # 1,n,c
            # 将输入特征 xi 作为查询向量（decoder_q），与编码器的输出 decoder_kv 进行解码：

            decoder_kv = self.transformer_encoder(encoder_qkv)

            recons = self.transformer_decoder(decoder_q, decoder_kv)
            # 将解码后的特征恢复到 [channel, height, width] 的格式
            recons = recons.squeeze(0).permute(1,0).reshape(c, h, w)

            gre.append(recons)
        # 将所有样本的重建特征图堆叠为一个 Tensor，shape 为 [batch_size, channel, height, width]：
        gre = torch.stack(gre,dim=0)

        return gre


class RegionAdaption(nn.Module):

    def __init__(self, channel):

        super(RegionAdaption, self).__init__()


        head = 4
        # dim=channel：特征的维度。
        # depth=1：Transformer 编码器的层数。
        # heads=head：多头注意力机制的头数。
        # dim_head=channel // head：每个头的特征维度。
        # mlp_dim=2*channel：前馈网络中的隐藏维度。
        self.transformer_encoder = Transformer(dim=channel, depth=1, heads = head,
                                       dim_head=channel // head,
                                       mlp_dim=2*channel, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim = channel, depth=1,
                                                      heads=head, dim_head= channel // head, mlp_dim=2*channel,
                                                      dropout=0,
                                                      softmax=True)


    def forward(self, sp, x, xp):
        # 超像素分割图（shape: h, w），表示每个像素所属的区域编号
        # 原始特征图（shape: c, h, w）。
        # 经过一定预处理（如 DWConv）的特征图（shape: c, h, w）

        c, h, w = x.shape

        # 将输入特征图x和xp转换成 1D 特征向量
        x = x.permute(1, 2, 0).reshape(-1, c)
        # reshape成 (h * w, c)
        xp = xp.permute(1, 2, 0).reshape(-1, c)

        sp_values = torch.unique(sp)
        # 使用 torch.unique(sp) 获取超像素图中所有唯一的区域编号：
        # 区域索引列表

        sp_num = sp_values.shape[0]
        # 总区域数

        x_r = torch.zeros(x.shape).cuda()
        # x_r：存储更新后的特征图，与输入 x 的形状相同。

        d_r = torch.zeros([sp_num,c]).cuda()
        # mean value of region 存储每个区域的平均特征描述符（sp_num, c）

        # 逐区域特征聚合
        for i in range(sp_num):

            region_pos_2d = torch.nonzero(sp == int(sp_values[i]),as_tuple=False)
            # 获取区域内的特征表示（x 和 xp 相加）。找出该区域的像素位置（1D 索引）
            region_pos_1d = region_pos_2d[:, 0] * w + region_pos_2d[:, 1]

            encoder_qkv = x[region_pos_1d] + xp[region_pos_1d]

            # decoder_q = x[region_pos_1d].unsqueeze(0)

            decoder_kv = self.transformer_encoder(encoder_qkv.unsqueeze(0))
            # 使用 Transformer 编码器处理该区域的特征。

            # recons = self.transformer_decoder(decoder_q, decoder_kv)
            # 将处理后的特征重新放回到对应的像素位置。
            x_r[region_pos_1d] = decoder_kv.squeeze(0)
            # 计算区域内像素特征的均值，并存储到 d_r 中。
            d_r[i] = torch.mean(x_r[region_pos_1d],dim=0)
        # 将处理后的特征图 x_r 还原成 (c, h, w) 格式：
        x_r = x_r.permute(1, 0).reshape(c, h, w)
        # x_r：增强后的特征图，shape 为 (c, h, w)。
        # d_r：每个区域的平均特征描述符，shape 为 (sp_num, c)。
        return x_r, d_r


class Posi_attn(nn.Module):

    def __init__(self, channel):

        super(Posi_attn,self).__init__()


        self.pos = DWConv(channel)
        # 对输入特征图进行卷积处理，提取局部空间信息

        self.ra = RegionAdaption(channel)
        # 自适应地提取输入特征图在空间上的局部区域特征。
        # 通过分割成区域并为每个区域计算描述符（特征聚合）。

        self.ga = GlobalAggregation(channel)
    #     聚合所有区域特征描述符，并生成全局特征图。

    def forward(self, x):

        b, _, x_h, x_w = x.shape
        # x：输入特征图，形状为 (b, c, h, w)。

        # np.save('F_feature.npy',x.squeeze(0).detach().cpu().numpy())

        xp1 = self.pos(x)
        # 使用DWConv提取局部特征

        _, sps, _ = ssn_iter(x, n_spixels=128, n_iter=10)
        # 通过 ssn_iter 函数将输入特征图 x 分割成 num_spixels 个超级像素区域，迭代次数为 10。
        # 返回分割结果 sps（超级像素图）

        # _, sps, _  = gen_HR(x, K=self.args.groups, n_iter=10)

        sps = sps + 1
        # # 确保超级像素标签从 1 开始

        sps = sps.reshape(b, x_h, x_w)
        # 超级像素图 sps 会被调整为 (b, h, w) 的形状。

        r_features = []

        r_descriptors = []

        # 对每个批次样本，使用 RegionAdaption 模块计算局部区域特征 r_feature 和区域描述符 r_descriptor
        for i in range(b):
            r_feature, r_descriptor = self.ra(sps[i], x[i], xp1[i])

            r_features.append(r_feature)
            # 每个区域对应的特征。

            r_descriptors.append(r_descriptor)
        #     r_descriptors：每个区域对应的描述符。

        # r_features = torch.stack(r_features, dim=0) #b,c,h,w

        # r_descriptors = torch.stack(r_descriptors, dim=0) #b,s,c

        # xp2 = self.pos2(r_descriptors.permute(0,2,1))

        # ra = torch.stack(r_features ,dim=0)
        # 将 r_features 堆叠成一个批量特征图

        # np.save('F_rac.npy',ra.squeeze(0).detach().cpu().numpy())

        g_features = self.ga(r_descriptors, r_features)
        # 使用 GlobalAggregation 模块，将区域描述符 r_descriptors 和区域特征 r_features 进一步聚合，生成最终的全局特征图。

        # np.save('F_gac.npy', g_features.squeeze(0).detach().cpu().numpy())

        return g_features