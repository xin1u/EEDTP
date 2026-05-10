"""Mixture-of-Experts (MoE) adapters with time-based prompts (Fig. 8d).

Each degradation type maps to a specific time step t, which routes
through a time-based router to select and weight NE experts.
Inserted in decoder layers of the IR network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEAdapter(nn.Module):
    """Single MoE adapter layer with time-based routing.

    Architecture (Fig. 8d):
        Time MLP -> Router_t -> weighted sum of NE expert outputs

    Each expert is a lightweight bottleneck: Conv1x1(down) -> GELU -> Conv1x1(up)

    Args:
        dim: feature channel dimension
        num_experts: number of experts (NE, default 10)
        T: number of possible time steps
        reduction: bottleneck reduction ratio
    """
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
        """Forward pass.

        Args:
            x: feature map [B, C, H, W]
            t: timestep (int or scalar tensor)
        Returns:
            x + scale * MoE_output
        """
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
    """Wraps MoE adapter for insertion into decoder blocks.

    Usage:
        Place after each decoder level's NAFBlocks to add
        task-specific adaptation for multi-task unified IR.
    """
    def __init__(self, dim, num_experts=10, T=50, reduction=4):
        super().__init__()
        self.adapter = MoEAdapter(dim, num_experts, T, reduction)

    def forward(self, x, t):
        return self.adapter(x, t)


def attach_moe_adapters(net, num_experts=10, T=50, reduction=4):
    """Attach MoE adapters to each decoder level of EEDTPRestorationNet.

    Args:
        net: EEDTPRestorationNet instance
        num_experts: number of experts per adapter
        T: max timestep
        reduction: bottleneck reduction
    Returns:
        list of MoEAdapterBlock modules (already registered on net)
    """
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
