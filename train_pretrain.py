
import argparse, os, time, random, logging, sys
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
from glob import glob

sys.path.append(os.getcwd())

from networks.eedtp_arch import EEDTPRestorationNet
from networks.diffusion_reg import NoiseSchedule, compute_pretrain_loss
from utils.dist_utils import init_dist, is_main_process, cleanup


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GTDataset(Dataset):
    """Dataset for loading GT images for diffusion pre-training."""
    def __init__(self, root_dir, crop_size=128, max_samples=None):
        super().__init__()
        self.crop_size = crop_size
        self.images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']:
            self.images.extend(glob(os.path.join(root_dir, '**', ext), recursive=True))
        self.images.sort()
        if max_samples and len(self.images) > max_samples:
            self.images = self.images[:max_samples]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = transforms.ToTensor()(img)

        _, h, w = img.shape
        if h < self.crop_size or w < self.crop_size:
            img = F.interpolate(img.unsqueeze(0), size=(self.crop_size, self.crop_size),
                                mode='bilinear', align_corners=False).squeeze(0)
        else:
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            img = img[:, top:top + self.crop_size, left:left + self.crop_size]

        if random.random() > 0.5:
            img = torch.flip(img, [-1])
        if random.random() > 0.5:
            img = torch.flip(img, [-2])

        return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', type=str, default='./data/gt_images/')
parser.add_argument('--save_path', type=str, default='./ckpt/')
parser.add_argument('--writer_dir', type=str, default='./tf-logs/')
parser.add_argument('--experiment_name', type=str, default='pretrain_eedtp')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--diffusion_T', type=int, default=50)
parser.add_argument('--total_iters', type=int, default=100000)

# arch
parser.add_argument('--base_channel', type=int, default=48)
parser.add_argument('--num_res', type=int, default=6)
parser.add_argument('--img_channel', type=int, default=3)
parser.add_argument('--enc_blks', nargs='+', type=int, default=[2, 2, 4, 28])
parser.add_argument('--dec_blks', nargs='+', type=int, default=[2, 2, 2, 2])

args = parser.parse_args()


if __name__ == '__main__':
    rank, local_rank, world_size = init_dist()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main_process():
        os.makedirs(args.save_path, exist_ok=True)
        writer = SummaryWriter(args.writer_dir + args.experiment_name)
        print('device:', device, f'(world_size={world_size})')

    net = EEDTPRestorationNet(
        img_channel=args.img_channel,
        width=args.base_channel,
        middle_blk_num=args.num_res,
        enc_blk_nums=args.enc_blks,
        dec_blk_nums=args.dec_blks,
    ).to(device)

    if world_size > 1:
        net = DDP(net, device_ids=[local_rank])
    net_without_ddp = net.module if world_size > 1 else net

    if is_main_process():
        print(f'#parameters: {sum(p.numel() for p in net.parameters()):,}')

    schedule = NoiseSchedule(max_sigma=10, T=args.diffusion_T, schedule='linear', eps=0.005, device=device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_iters, eta_min=1e-7)

    train_dataset = GTDataset(args.gt_dir, crop_size=args.crop_size)
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=8, drop_last=True)

    if is_main_process():
        print(f'GTDataset: {len(train_dataset)} images')
        print(f'len(train_loader): {len(train_loader)}')
        logging.basicConfig(
            filename=os.path.join(args.save_path, args.experiment_name + '.log'),
            level=logging.INFO)

    global_step = 0

    for epoch in range(args.epochs):
        net.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        st = time.time()

        for i, x0 in enumerate(train_loader):
            x0 = x0.to(device)

            optimizer.zero_grad()
            loss = compute_pretrain_loss(net, x0, schedule)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1

            if is_main_process() and global_step % 100 == 0:
                writer.add_scalar('pretrain/loss', loss.item(), global_step)
                print(f"epoch:{epoch} [{i+1}/{len(train_loader)}] "
                      f"iter:{global_step} lr:{optimizer.param_groups[0]['lr']:.7f} "
                      f"loss:{loss.item():.5f} t:{time.time()-st:.1f}s")
                logging.info(f"epoch:{epoch} iter:{global_step} loss:{loss.item():.5f}")
                st = time.time()

            if is_main_process() and global_step % 10000 == 0:
                torch.save(net_without_ddp.state_dict(),
                           os.path.join(args.save_path, 'pretrained_model.pth'))

            if global_step >= args.total_iters:
                break

        if global_step >= args.total_iters:
            break

        if is_main_process():
            torch.save(net_without_ddp.state_dict(),
                       os.path.join(args.save_path, 'latest_pretrain.pth'))

    if is_main_process():
        torch.save(net_without_ddp.state_dict(),
                   os.path.join(args.save_path, 'pretrained_model.pth'))
        print(f'Pre-training finished at iter {global_step}')
        writer.close()

    cleanup()
