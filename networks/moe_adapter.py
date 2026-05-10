
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEAdapter(nn.Module):

    def __init__(self, dim, num_experts=10, T=50, reduction=4):
        super().__init__()
        self.num_experts = num_experts
        hidden = max(dim // reduction, 16)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, hidden, 1),
                nn.GELU(),
                nn.Conv2d(hidden, dim, 1),
            )
            for _ in range(num_experts)
        ])

        self.routers = nn.ModuleDict()
        for t in range(1, T + 1):
            self.routers[str(t)] = nn.Linear(dim, num_experts)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, t):

        t_key = str(int(t))
        if t_key not in self.routers:
            t_key = str(max(1, min(int(t), len(self.routers))))

        feat = self.gap(x).squeeze(-1).squeeze(-1)  # [B, C]
        weights = F.softmax(self.routers[t_key](feat), dim=-1)  # [B, NE]

        expert_out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert_out + weights[:, i:i+1, None, None] * expert(x)

        return x + self.scale * expert_out


class MoEAdapterBlock(nn.Module):

    def __init__(self, dim, num_experts=10, T=50, reduction=4):
        super().__init__()
        self.adapter = MoEAdapter(dim, num_experts, T, reduction)

    def forward(self, x, t):
        return self.adapter(x, t)


def attach_moe_adapters(net, num_experts=10, T=50, reduction=4):

    adapters = nn.ModuleList()
    chan = net.intro.out_channels
    for _ in net.encoders:
        chan = chan * 2
    for i, dec_blks in enumerate(net.decoders):
        chan = chan // 2
        adapter = MoEAdapterBlock(chan, num_experts, T, reduction)
        adapters.append(adapter)

    net.moe_adapters = adapters
    return adapters


if __name__ == '__main__':
    adapter = MoEAdapter(dim=64, num_experts=10, T=50, reduction=4)
    x = torch.randn(2, 64, 32, 32)
    out = adapter(x, t=25)
    print(f'MoEAdapter: input {x.shape} -> output {out.shape}')
    print(f'adapter params: {sum(p.numel() for p in adapter.parameters()):,}')
