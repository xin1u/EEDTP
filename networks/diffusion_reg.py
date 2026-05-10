 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class NoiseSchedule:
 
    def __init__(self, max_sigma=10, T=50, schedule='linear', eps=0.005, device=None):
        self.T = T
        self.device = device
        self.max_sigma = max_sigma / 255 if max_sigma >= 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule, eps)

    def _initialize(self, max_sigma, T, schedule, eps=0.005):
        if schedule == 'cosine':
            steps = T + 2
            x = torch.linspace(0, T + 1, steps + 1, dtype=torch.float32)
            alphas = torch.cos(((x / (T + 1)) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas = alphas / alphas[0]
            thetas = 1 - alphas[1:-1]
        else:
            scale = 1000 / (T + 1)
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            thetas = torch.linspace(beta_start, beta_end, T + 1, dtype=torch.float32)

        thetas_cumsum = torch.cumsum(thetas, dim=0) - thetas[0]
        dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = torch.sqrt(max_sigma ** 2 * (1 - torch.exp(-2 * thetas_cumsum * dt)))

        self.sigma_bars = sigma_bars

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def to(self, device):
        self.sigma_bars = self.sigma_bars.to(device)
        self.device = device
        return self


# ---------------------------------------------------------------------------
# Pre-training denoising loss  
# ---------------------------------------------------------------------------

def compute_pretrain_loss(net, x0, schedule):
 
    batch = x0.shape[0]
    device = x0.device

    timesteps = torch.randint(1, schedule.T + 1, (batch,)).long()
    sigma = schedule.sigma_bar(timesteps.view(-1, 1, 1, 1)).to(device)
    noise = torch.randn_like(x0)
    x_t = x0 + sigma * noise

    residual_pred = net(x_t, timesteps.to(device))
    restored = x_t - residual_pred
    return F.l1_loss(restored, x0)


# ---------------------------------------------------------------------------
# Fine-tuning denoising loss 
# ---------------------------------------------------------------------------

def compute_denoising_loss(net, x0, schedule):
 
    return compute_pretrain_loss(net, x0, schedule)


# ---------------------------------------------------------------------------
# Parameter importance regularization  
# ---------------------------------------------------------------------------

class ParameterRegularizer:
 
    def __init__(self, net):
        self.theta0 = {}
        self.omega = {}
        for name, p in net.named_parameters():
            self.theta0[name] = p.data.clone()
            self.omega[name] = torch.zeros_like(p.data)

    def compute_importance(self, net, dataloader, schedule, device, num_batches=100):
        """Estimate importance weights from pre-training gradients."""
        grad_sum = {n: torch.zeros_like(p) for n, p in net.named_parameters()}
        count = 0

        net.train()
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            if isinstance(batch, (list, tuple)):
                gt = batch[1].to(device) if len(batch) > 1 else batch[0].to(device)
            else:
                gt = batch.to(device)

            net.zero_grad()
            loss = compute_pretrain_loss(net, gt, schedule)
            loss.backward()

            for name, p in net.named_parameters():
                if p.grad is not None:
                    grad_sum[name] += p.grad.data.abs()
            count += 1

        for name in grad_sum:
            self.omega[name] = grad_sum[name] / max(count, 1)

        net.zero_grad()

    def loss(self, net, lambda_reg=0.2):
        """Compute L_reg (Eq. 5)."""
        reg = torch.tensor(0., device=next(net.parameters()).device)
        for name, p in net.named_parameters():
            if name not in self.theta0:
                continue
            delta = p - self.theta0[name].to(p.device)
            omega = self.omega[name].to(p.device)
            reg = reg + (omega * delta.abs() + 0.5 * omega ** 2 * delta ** 2).sum()
        return lambda_reg * reg


# ---------------------------------------------------------------------------
# Gradient orthogonality loss  
# ---------------------------------------------------------------------------

class GradientOrthogonalLoss:
 
    @staticmethod
    def compute(net, loss_gen, loss_res):
        params = [p for p in net.parameters() if p.requires_grad]

        grads_gen = torch.autograd.grad(loss_gen, params, retain_graph=True,
                                        allow_unused=True)
        grads_res = torch.autograd.grad(loss_res, params, retain_graph=True,
                                        allow_unused=True)

        g_gen_parts = []
        g_res_parts = []
        for g_gen, g_res in zip(grads_gen, grads_res):
            if g_gen is None or g_res is None:
                continue
            g_gen_parts.append(g_gen.flatten())
            g_res_parts.append(g_res.flatten())

        if len(g_gen_parts) == 0:
            return torch.tensor(0., device=params[0].device)

        g_gen_vec = torch.cat(g_gen_parts)
        g_res_vec = torch.cat(g_res_parts)

        s = F.cosine_similarity(g_gen_vec.unsqueeze(0), g_res_vec.unsqueeze(0))

        return 1.0 - s.squeeze()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from eedtp_arch import EEDTPRestorationNet

    device = torch.device('cpu')
    schedule = NoiseSchedule(max_sigma=10, T=50, schedule='linear', eps=0.005)

    net = EEDTPRestorationNet(
        img_channel=3, width=16, middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 2], dec_blk_nums=[1, 1, 1, 1],
    )
    print(f'params: {sum(p.numel() for p in net.parameters()):,}')

    x0 = torch.rand(2, 3, 64, 64)
    loss_pt = compute_pretrain_loss(net, x0, schedule)
    print(f'pretrain loss: {loss_pt.item():.4f}')

    loss_ft = compute_denoising_loss(net, x0, schedule)
    print(f'denoising loss: {loss_ft.item():.4f}')

    param_reg = ParameterRegularizer(net)
    l_reg = param_reg.loss(net)
    print(f'L_reg (before training, should be ~0): {l_reg.item():.6f}')

    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    opt.zero_grad()
    loss_pt.backward()
    opt.step()

    l_reg_after = param_reg.loss(net)
    print(f'L_reg (after step, should be > 0): {l_reg_after.item():.6f}')

    # gradient orthogonal loss
    x_lq = torch.rand(2, 3, 64, 64)
    x_gt = torch.rand(2, 3, 64, 64)

    residual = net(x_lq, time=25)
    loss_res = F.l1_loss(x_lq - residual, x_gt)
    loss_gen = compute_pretrain_loss(net, x_gt, schedule)

    l_orthog = GradientOrthogonalLoss.compute(net, loss_gen, loss_res)
    print(f'L_orthog: {l_orthog.item():.4f}')
