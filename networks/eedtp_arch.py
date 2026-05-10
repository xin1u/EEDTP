"""EEDTP restoration network with time-step conditioning (Fig. 8d).

Based on X-Restormer / NAFNet backbone with:
  - SimpleGate (SGM) + SCA channel attention
  - Sinusoidal time embedding via linear layer
  - AdaLN-style time conditioning in each block
  - Conditional input: x = cat(inp - cond, cond) -> 2*img_channel input
  - Output is noise/residual: restored = cond - net_output
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, inp_pair):
        """Forward pass. Input is (feature, time_emb) tuple for nn.Sequential compatibility."""
        inp, time = inp_pair

        x = self.norm1(inp)
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)
        x = x * (scale_att + 1) + shift_att

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn

        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        x = y + x * self.gamma

        return x, time


class EEDTPRestorationNet(nn.Module):
    """EEDTP conditional restoration network (Sec. III-E, Fig. 8d).

    Conditional NAFNet U-Net backbone with time-step conditioning.
    Input: (x_t, cond, time) where cond=LQ acts as the SDE mean (mu).
    The network takes cat(x_t - cond, cond) as input and predicts noise.
    Restoration: output = cond - net(x_t, cond, time).

    During inference, x_t = cond = LQ, so the network predicts the
    degradation residual directly.

    Args:
        img_channel: input/output image channels (default 3)
        width: base channel width (default 32)
        middle_blk_num: number of middle blocks (default 6)
        enc_blk_nums: list of encoder block counts per level
        dec_blk_nums: list of decoder block counts per level
    """
    def __init__(self, img_channel=3, width=32, middle_blk_num=6,
                 enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()

        fourier_dim = width
        time_dim = width * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim * 2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim),
        )

        self.intro = nn.Conv2d(img_channel * 2, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan, time_dim) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan, time_dim) for _ in range(num)])
            )

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp, cond, time):
        """Forward pass.

        Args:
            inp: noisy state x_t [B, C, H, W] (during training) or LQ (during inference)
            cond: condition image (LQ) [B, C, H, W], acts as SDE mean mu
            time: diffusion timestep, int/float or [B] tensor
        Returns:
            predicted noise [B, C, H, W]
        """
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(inp.device)

        x = inp - cond
        x = torch.cat([x, cond], dim=1)

        t = self.time_mlp(time)

        B, C, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x, _ = encoder([x, t])
            encs.append(x)
            x = down(x)

        x, _ = self.middle_blks([x, t])

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, t])

        x = self.ending(x)
        x = x[..., :H, :W]

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == '__main__':
    device = torch.device('cpu')
    net = EEDTPRestorationNet(
        img_channel=3, width=32, middle_blk_num=6,
        enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
    )
    params = sum(p.numel() for p in net.parameters())
    print(f'params: {params:,}')

    x = torch.randn(1, 3, 128, 128)
    cond = torch.randn(1, 3, 128, 128)

    # test with timestep
    noise = net(x, cond, time=25)
    print(f'with t=25: input {x.shape} -> noise {noise.shape}')
    restored = cond - noise
    print(f'restored: {restored.shape}')

    # test with batch timesteps
    t_batch = torch.tensor([10, 20])
    x2 = torch.randn(2, 3, 64, 64)
    cond2 = torch.randn(2, 3, 64, 64)
    noise2 = net(x2, cond2, time=t_batch)
    print(f'batch t: input {x2.shape} -> noise {noise2.shape}')
