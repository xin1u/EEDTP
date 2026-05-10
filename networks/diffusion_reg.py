"""Diffusion regularization for EEDTP (Sec. III-D).

IRSDE noise perturbation (conditioned on LQ as mean) plus regularization:
  - L_reg:     parameter importance regularization (Eq. 5)
  - L_orthog:  gradient orthogonality loss (Eq. 6-8)
  - w_decay:   layer-wise weight decay for denoising gradients (Eq. 10)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# ---------------------------------------------------------------------------
# IRSDE (Image Restoration SDE, conditioned on LQ)
# ---------------------------------------------------------------------------

class IRSDE:
    """Image Restoration SDE with LQ as drift target (mu).

    Forward process: q(x_t | x_0, mu) = N(mu_bar(x_0, t), sigma_bar(t))
    where mu_bar = mu + (x_0 - mu) * exp(-theta_cumsum * dt)

    The forward process degrades GT toward LQ (not toward pure noise),
    making the denoising objective naturally aligned with restoration.
    """
    def __init__(self, max_sigma=10, T=50, schedule='linear', eps=0.005, device=None):
        self.T = T
        self.device = device
        self.max_sigma = max_sigma / 255 if max_sigma >= 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule, eps)
        self.mu = 0.
        self.model = None

    def _initialize(self, max_sigma, T, schedule, eps=0.005):
        def linear_theta_schedule(timesteps):
            timesteps = timesteps + 1
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_theta_schedule(timesteps, s=0.008):
            timesteps = timesteps + 2
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        else:
            thetas = linear_theta_schedule(T)

        sigmas = torch.sqrt(max_sigma ** 2 * 2 * thetas)
        thetas_cumsum = torch.cumsum(thetas, dim=0) - thetas[0]
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = torch.sqrt(max_sigma ** 2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))

        self.thetas = thetas
        self.sigmas = sigmas
        self.thetas_cumsum = thetas_cumsum
        self.sigma_bars = sigma_bars

    def _to(self, device):
        self.thetas = self.thetas.to(device)
        self.sigmas = self.sigmas.to(device)
        self.thetas_cumsum = self.thetas_cumsum.to(device)
        self.sigma_bars = self.sigma_bars.to(device)
        self.device = device

    def set_mu(self, mu):
        self.mu = mu

    def set_model(self, model):
        self.model = model

    def mu_bar(self, x0, t):
        return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def noise_fn(self, x, t, **kwargs):
        return self.model(x, self.mu, t, **kwargs)

    def generate_random_states(self, x0, mu):
        """Generate noisy states from GT (x0) and LQ (mu).

        Returns:
            timesteps: [B,1,1,1] random timesteps
            noisy_states: x_t sampled from q(x_t | x0, mu)
        """
        if self.device is not None:
            x0 = x0.to(self.device)
            mu = mu.to(self.device)

        self.set_mu(mu)
        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()
        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def reverse_sde_step_mean(self, x, score, t):
        return x - (self.thetas[t] * (self.mu - x) - self.sigmas[t] ** 2 * score) * self.dt

    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t - 1] * self.dt)

        term1 = A * (1 - C ** 2) / (1 - B ** 2)
        term2 = C * (1 - A ** 2) / (1 - B ** 2)

        return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu

    def noise_state(self, tensor):
        return tensor + torch.randn_like(tensor) * self.max_sigma

    def weights(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt)


# ---------------------------------------------------------------------------
# Pre-training denoising loss (TMDDP, Sec. III-E.1)
# ---------------------------------------------------------------------------

def compute_pretrain_loss(net, x0, sde):
    """Denoising pre-training loss (TMDDP). GT images only, no LQ needed.

    Adds noise to GT: x_t = x_0 + sigma_bar(t) * noise.
    Network receives (x_t, x_t, t) -- cond is the noisy image itself,
    so input = cat(0, x_noisy). The network must learn to denoise x_t.

    This input pattern matches inference cat(0, LQ), ensuring the
    denoising prior transfers directly to restoration fine-tuning.

    Args:
        net: EEDTPRestorationNet
        x0: clean GT images [B, C, H, W]
        sde: IRSDE instance (uses sigma_bar for noise schedule)
    """
    batch = x0.shape[0]
    device = x0.device

    timesteps = torch.randint(1, sde.T + 1, (batch, 1, 1, 1)).long()
    sigma = sde.sigma_bar(timesteps).to(device)
    noise = torch.randn_like(x0)
    x_t = x0 + sigma * noise

    noise_pred = net(x_t, x_t, timesteps.squeeze())
    restored = x_t - noise_pred
    return F.l1_loss(restored, x0)


# ---------------------------------------------------------------------------
# Fine-tuning denoising loss (mixed training, Sec. III-E.2)
# ---------------------------------------------------------------------------

def compute_denoising_loss(net, x0, mu, sde):
    """Mixed denoising loss for fine-tuning. Uses IRSDE with mu=LQ.

    IRSDE forward: x_t ~ N(mu_bar(x0, t), sigma_bar(t)) where mu = LQ.
    Network receives (x_t, LQ, t), input = cat(x_t - LQ, LQ).
    Loss: L1(GT, LQ - noise_pred), aligned with restoration objective.

    Args:
        net: EEDTPRestorationNet
        x0: clean GT images [B, C, H, W]
        mu: LQ condition images [B, C, H, W]
        sde: IRSDE instance
    """
    sde.set_mu(mu)
    timesteps, noisy_states = sde.generate_random_states(x0, mu)

    noise_pred = net(noisy_states, mu, timesteps.squeeze())
    restored = mu - noise_pred
    return F.l1_loss(restored, x0)


# ---------------------------------------------------------------------------
# Parameter importance regularization (Eq. 3-5)
# ---------------------------------------------------------------------------

class ParameterRegularizer:
    """Parameter importance regularization (Sec. III-D, Eq. 5).

    Stores pre-trained parameters theta_0 and computes importance weights
    Omega from gradient accumulation. The regularization loss:
        L_reg = lambda * sum_k [ Omega_k * |delta_theta_k| + 0.5 * Omega_k^2 * delta_theta_k^2 ]
    """
    def __init__(self, net):
        self.theta0 = {}
        self.omega = {}
        for name, p in net.named_parameters():
            self.theta0[name] = p.data.clone()
            self.omega[name] = torch.zeros_like(p.data)

    def compute_importance(self, net, dataloader, sde, device, num_batches=100):
        """Estimate importance weights from pre-training gradients (Eq. 3-4).

        Omega_k = E[ |grad_k| ] (first-order importance approximation)
        Uses pretrain loss (not fine-tuning loss) to compute importance.
        """
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
            loss = compute_pretrain_loss(net, gt, sde)
            loss.backward()

            for name, p in net.named_parameters():
                if p.grad is not None:
                    grad_sum[name] += p.grad.data.abs()
            count += 1

        for name in grad_sum:
            self.omega[name] = grad_sum[name] / max(count, 1)

        net.zero_grad()

    def loss(self, net, lambda_reg=0.2):
        """Compute L_reg (Eq. 5).

        L_reg = lambda * sum_k [ Omega_k * |delta_k| + 0.5 * Omega_k^2 * delta_k^2 ]
        """
        reg = torch.tensor(0., device=next(net.parameters()).device)
        for name, p in net.named_parameters():
            delta = p - self.theta0[name].to(p.device)
            omega = self.omega[name].to(p.device)
            reg = reg + (omega * delta.abs() + 0.5 * omega ** 2 * delta ** 2).sum()
        return lambda_reg * reg


# ---------------------------------------------------------------------------
# Gradient orthogonality loss (Eq. 6-8)
# ---------------------------------------------------------------------------

class GradientOrthogonalLoss:
    """Gradient orthogonality between denoising and reconstruction (Sec. III-D, Eq. 6-8).

    s = avg cosine_sim(g_gen, g_res)   (cross-task, Eq. 6)
    d = avg cosine_sim within tasks     (intra-task, Eq. 7)
    L_orthog = (1 - s) + |d|           (Eq. 8)
    """
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
# Layer-wise weight decay for denoising gradients (Eq. 10)
# ---------------------------------------------------------------------------

def compute_layer_weight_decay(layer_idx, total_layers, tmat, a=0.05):
    """Compute weight decay coefficient for denoising gradient update (Eq. 10).

    w_decay(t) = exp(-a * t)

    Applied layer-wise: shallower layers get stronger denoising gradient,
    deeper layers get weaker.

    Args:
        layer_idx: current layer index (0 = shallowest)
        total_layers: total number of layers
        tmat: matching timestep for this degradation
        a: decay rate (default 0.05)
    """
    t_normalized = layer_idx / max(total_layers - 1, 1)
    return math.exp(-a * tmat * t_normalized)


def apply_denoising_weight_decay(net, denoising_grads, tmat, a=0.05):
    """Apply layer-wise weight decay to denoising gradients (Eq. 10).

    Shallow layers get full denoising gradient, deeper layers get
    exponentially decayed gradient.

    Args:
        net: EEDTPRestorationNet
        denoising_grads: dict of {name: grad_tensor}
        tmat: matching timestep
        a: decay rate
    """
    all_params = list(net.named_parameters())
    total = len(all_params)

    weighted_grads = {}
    for i, (name, p) in enumerate(all_params):
        if name in denoising_grads and denoising_grads[name] is not None:
            w = math.exp(-a * tmat * (i / max(total - 1, 1)))
            weighted_grads[name] = denoising_grads[name] * w
        else:
            weighted_grads[name] = None

    return weighted_grads


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from eedtp_arch import EEDTPRestorationNet

    device = torch.device('cpu')
    sde = IRSDE(max_sigma=10, T=50, schedule='linear', eps=0.005)

    net = EEDTPRestorationNet(
        img_channel=3, width=16, middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 2], dec_blk_nums=[1, 1, 1, 1],
    )
    print(f'params: {sum(p.numel() for p in net.parameters()):,}')

    # pre-training loss (should be non-trivial)
    x0 = torch.rand(2, 3, 64, 64)
    loss_pt = compute_pretrain_loss(net, x0, sde)
    print(f'pretrain loss: {loss_pt.item():.4f}')

    # fine-tuning denoising loss
    mu = torch.rand(2, 3, 64, 64)
    loss_ft = compute_denoising_loss(net, x0, mu, sde)
    print(f'finetune denoising loss: {loss_ft.item():.4f}')

    # parameter regularizer
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

    noise_pred = net(x_lq, x_lq, time=25)
    loss_res = F.l1_loss(x_lq - noise_pred, x_gt)
    loss_gen = compute_pretrain_loss(net, x_gt, sde)

    l_orthog = GradientOrthogonalLoss.compute(net, loss_gen, loss_res)
    print(f'L_orthog: {l_orthog.item():.4f}')

    # weight decay
    w = compute_layer_weight_decay(0, 10, tmat=25)
    print(f'weight decay (shallow, tmat=25): {w:.4f}')
    w = compute_layer_weight_decay(9, 10, tmat=25)
    print(f'weight decay (deep, tmat=25): {w:.4f}')
