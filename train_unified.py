
import time, torchvision, argparse, logging, sys, os, gc
import torch, random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.UTILS1 import compute_psnr
from utils.UTILS import AverageMeters
from torch.utils.tensorboard import SummaryWriter
from datasets.datasets_pairs import my_dataset, my_dataset_eval
from networks.eedtp_arch import EEDTPRestorationNet
from networks.moe_adapter import attach_moe_adapters
from networks.diffusion_reg import (
    IRSDE, compute_denoising_loss,
    ParameterRegularizer, GradientOrthogonalLoss,
)
from utils.dist_utils import init_dist, is_main_process, cleanup

sys.path.append(os.getcwd())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)


# tmat values from Table I, sorted for incremental training
TASK_CONFIG = [
    {'name': 'noisy',     'tmat': 4,  'tmat_frac': 0.08},
    {'name': 'rainy',     'tmat': 8,  'tmat_frac': 0.16},
    {'name': 'jpeg',      'tmat': 12, 'tmat_frac': 0.24},
    {'name': 'snowy',     'tmat': 15, 'tmat_frac': 0.30},
    {'name': 'inpainting','tmat': 19, 'tmat_frac': 0.38},
    {'name': 'raindrop',  'tmat': 22, 'tmat_frac': 0.44},
    {'name': 'shadowed',  'tmat': 27, 'tmat_frac': 0.54},
    {'name': 'lowlight',  'tmat': 38, 'tmat_frac': 0.76},
    {'name': 'hazy',      'tmat': 47, 'tmat_frac': 0.94},
    {'name': 'blurry',    'tmat': 50, 'tmat_frac': 1.00},
]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

# path
parser.add_argument('--experiment_name', type=str, default="train_unified_eedtp")
parser.add_argument('--unified_path', type=str, default='./experiments/')
parser.add_argument('--data_root', type=str, default='./data/')
parser.add_argument('--writer_dir', type=str, default='./tf-logs/')

# training
parser.add_argument('--iters_per_task', type=int, default=100000)
parser.add_argument('--BATCH_SIZE', type=int, default=16)
parser.add_argument('--Crop_patches', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--print_frequency', type=int, default=200)
parser.add_argument('--val_frequency', type=int, default=5000)
parser.add_argument('--fix_sampleA', type=int, default=30000)

# arch
parser.add_argument('--base_channel', type=int, default=32)
parser.add_argument('--num_res', type=int, default=6)
parser.add_argument('--img_channel', type=int, default=3)
parser.add_argument('--enc_blks', nargs='+', type=int, default=[1, 1, 1, 28])
parser.add_argument('--dec_blks', nargs='+', type=int, default=[1, 1, 1, 1])

# multi-task
parser.add_argument('--tasks', type=str,
                    default='noisy,rainy,jpeg,snowy,inpainting,raindrop,shadowed,lowlight,hazy,blurry')
parser.add_argument('--num_experts', type=int, default=10)

# load pre-trained model
parser.add_argument('--pre_model', type=str, default='./ckpt/pretrained_model.pth')

# diffusion regularization
parser.add_argument('--diffusion_T', type=int, default=50)
parser.add_argument('--lambda_reg', type=float, default=0.2)
parser.add_argument('--gen_prob', type=float, default=0.1)

args = parser.parse_args()

task_names = [t.strip() for t in args.tasks.split(',')]
task_order = [tc for tc in TASK_CONFIG if tc['name'] in task_names]
task_order.sort(key=lambda x: x['tmat'])

trans_eval = transforms.Compose([transforms.ToTensor()])


def get_task_dataset(task_name, split='train'):
    in_path = os.path.join(args.data_root, task_name, f'{split}_input')
    gt_path = os.path.join(args.data_root, task_name, f'{split}_gt')

    if not os.path.exists(in_path):
        in_path = os.path.join(args.data_root, task_name, 'input')
        gt_path = os.path.join(args.data_root, task_name, 'gt')

    if split == 'train':
        return my_dataset(root_in=in_path, root_label=gt_path,
                          crop_size=args.Crop_patches, fix_sample_A=args.fix_sampleA)
    else:
        return my_dataset_eval(root_in=in_path, root_label=gt_path,
                               transform=trans_eval, fix_sample=500)


def test_unified(net, task_order, device, Dname='val'):
    net.eval()
    net_eval = net.module if isinstance(net, DDP) else net
    results = {}
    with torch.no_grad():
        for tc in task_order:
            try:
                eval_dataset = get_task_dataset(tc['name'], split='val')
                eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=4)
            except Exception:
                continue
            psnr_sum = 0
            count = 0
            for data_in, label, name in eval_loader:
                inputs = Variable(data_in).to(device)
                labels = Variable(label).to(device)
                noise_pred = net_eval(inputs, inputs, tc['tmat'])
                outputs = inputs - noise_pred
                psnr_sum += compute_psnr(outputs, labels)
                count += 1
            if count > 0:
                results[tc['name']] = psnr_sum / count

    for task, psnr in results.items():
        print(f"  [{Dname}] {task}: {psnr:.2f}")
        logging.info(f"  [{Dname}] {task}: {psnr:.2f}")

    if results:
        avg = sum(results.values()) / len(results)
        print(f"  [{Dname}] average: {avg:.2f}")
    return results


if __name__ == '__main__':
    rank, local_rank, world_size = init_dist()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    exper_name = args.experiment_name
    SAVE_PATH = args.unified_path + exper_name + '/'

    if is_main_process():
        os.makedirs(SAVE_PATH, exist_ok=True)
        writer = SummaryWriter(args.writer_dir + exper_name)
        logging.basicConfig(filename=SAVE_PATH + exper_name + '.log', level=logging.INFO)
        print(f'device: {device} (world_size={world_size})')

    net = EEDTPRestorationNet(
        img_channel=args.img_channel,
        width=args.base_channel,
        middle_blk_num=args.num_res,
        enc_blk_nums=args.enc_blks,
        dec_blk_nums=args.dec_blks,
    )

    if args.pre_model and os.path.exists(args.pre_model):
        net.load_state_dict(torch.load(args.pre_model, map_location='cpu'), strict=True)
        if is_main_process():
            print('loaded pre-trained model from', args.pre_model)

    adapters = attach_moe_adapters(net, num_experts=args.num_experts, T=args.diffusion_T)
    net.to(device)

    if world_size > 1:
        net = DDP(net, device_ids=[local_rank], find_unused_parameters=True)
    net_without_ddp = net.module if world_size > 1 else net

    if is_main_process():
        print(f'#parameters (with MoE): {sum(p.numel() for p in net.parameters()):,}')

    sde = IRSDE(max_sigma=10, T=args.diffusion_T, schedule='linear', eps=0.005, device=device)

    global_step = 0
    accumulated_tasks = []

    for step_idx, tc in enumerate(task_order):
        task_name = tc['name']
        tmat = tc['tmat']
        accumulated_tasks.append(tc)

        if is_main_process():
            print(f'\n{"="*60}')
            print(f'Incremental step {step_idx+1}/{len(task_order)}: {task_name} (tmat={tmat})')
            print(f'Accumulated tasks: {[t["name"] for t in accumulated_tasks]}')
            logging.info(f'Step {step_idx+1}: {task_name} tmat={tmat}')

        param_reg = ParameterRegularizer(net_without_ddp)

        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
        scheduler = CosineAnnealingLR(optimizer, T_max=args.iters_per_task, eta_min=1e-7)

        try:
            train_dataset = get_task_dataset(task_name, split='train')
        except Exception as e:
            if is_main_process():
                print(f'skip {task_name}: {e}')
            continue

        train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
        train_loader = DataLoader(
            train_dataset, batch_size=args.BATCH_SIZE,
            num_workers=8, shuffle=(train_sampler is None),
            sampler=train_sampler, drop_last=True)

        train_meters = AverageMeters()
        step = 0
        epoch = 0
        st = time.time()

        while step < args.iters_per_task:
            net.train()
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for train_data in train_loader:
                data_in, label, img_name = train_data
                step += 1
                global_step += 1

                inputs = Variable(data_in).to(device)
                labels = Variable(label).to(device)

                noise_pred = net(inputs, inputs, tmat)
                outputs = inputs - noise_pred
                loss_content = F.l1_loss(outputs, labels)
                total_loss = loss_content

                loss_gen = torch.tensor(0.)
                loss_orthog = torch.tensor(0.)
                loss_reg = torch.tensor(0.)

                if random.random() < args.gen_prob:
                    loss_gen = compute_denoising_loss(net, labels, inputs, sde)
                    loss_orthog = GradientOrthogonalLoss.compute(net, loss_gen, loss_content)
                    loss_reg = param_reg.loss(net_without_ddp, lambda_reg=args.lambda_reg)
                    total_loss = total_loss + loss_reg + loss_orthog

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                if is_main_process() and step % args.print_frequency == 0:
                    out_psnr = compute_psnr(outputs, labels)
                    print(f"  [{task_name}] step:{step}/{args.iters_per_task} "
                          f"loss:{total_loss.item():.5f} psnr:{out_psnr:.2f} t:{time.time()-st:.1f}s")
                    st = time.time()

                if step >= args.iters_per_task:
                    break

        if is_main_process():
            ckpt_path = os.path.join(SAVE_PATH, f'model_step{step_idx+1}_{task_name}.pth')
            torch.save(net_without_ddp.state_dict(), ckpt_path)
            print(f'saved: {ckpt_path}')
            test_unified(net, accumulated_tasks, device)

    if is_main_process():
        torch.save(net_without_ddp.state_dict(), os.path.join(SAVE_PATH, 'unified_model.pth'))
        print('\nFinal evaluation:')
        test_unified(net, task_order, device)
        print('Unified training finished.')
        writer.close()

    cleanup()
