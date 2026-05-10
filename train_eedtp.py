
import time, torchvision, argparse, logging, sys, os, gc
import torch, random
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.UTILS1 import compute_psnr
from utils.UTILS import AverageMeters, print_args_parameters
from loss.losses import fftLoss
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from datasets.datasets_pairs import my_dataset, my_dataset_eval, my_dataset_wTxt
from networks.eedtp_arch import EEDTPRestorationNet
from networks.diffusion_reg import (
    NoiseSchedule, compute_pretrain_loss, compute_denoising_loss,
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

setup_seed(20)


TMAT_TABLE = {
    'denoising': 4, 'noisy': 4,
    'deraining': 8, 'rainy': 8,
    'jpeg': 12,
    'desnowing': 15, 'snowy': 15,
    'inpainting': 19,
    'raindrop': 22,
    'shadow': 27, 'shadowed': 27, 'deshadow': 27,
    'lowlight': 38, 'low-light': 38, 'low_light': 38,
    'dehazing': 47, 'hazy': 47,
    'deblurring': 50, 'blurry': 50,
}


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
parser.add_argument('--experiment_name', type=str, default="train_eedtp_gef")
parser.add_argument('--unified_path', type=str, default='./experiments/')
parser.add_argument('--training_in_path', type=str, default='./data/train_input/')
parser.add_argument('--training_gt_path', type=str, default='./data/train_gt/')
parser.add_argument('--training_path_txt', nargs='*', default=None)
parser.add_argument('--writer_dir', type=str, default='./tf-logs/')
parser.add_argument('--eval_in_path', type=str, default='./data/val_input/')
parser.add_argument('--eval_gt_path', type=str, default='./data/val_gt/')

# training
parser.add_argument('--total_iters', type=int, default=500000)
parser.add_argument('--BATCH_SIZE', type=int, default=16)
parser.add_argument('--Crop_patches', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--print_frequency', type=int, default=200)
parser.add_argument('--val_frequency', type=int, default=5000)
parser.add_argument('--save_frequency', type=int, default=10000)
parser.add_argument('--max_psnr', type=int, default=40)
parser.add_argument('--fix_sampleA', type=int, default=30000)

# arch
parser.add_argument('--base_channel', type=int, default=48)
parser.add_argument('--num_res', type=int, default=6)
parser.add_argument('--img_channel', type=int, default=3)
parser.add_argument('--enc_blks', nargs='+', type=int, default=[2, 2, 4, 28])
parser.add_argument('--dec_blks', nargs='+', type=int, default=[2, 2, 2, 2])

# task
parser.add_argument('--task', type=str, default='dehazing')
parser.add_argument('--tmat', type=int, default=None)

# load pre-trained model
parser.add_argument('--load_pre_model', type=str2bool, default=True)
parser.add_argument('--pre_model', type=str, default='./ckpt/pretrained_model.pth')

# diffusion regularization
parser.add_argument('--diffusion_T', type=int, default=50)
parser.add_argument('--lambda_reg', type=float, default=0.2)
parser.add_argument('--gen_prob', type=float, default=0.1)
parser.add_argument('--importance_batches', type=int, default=50)
parser.add_argument('--lambda_fft', type=float, default=0.1)

args = parser.parse_args()

if args.tmat is None:
    args.tmat = TMAT_TABLE.get(args.task.lower(), 25)

trans_eval = transforms.Compose([transforms.ToTensor()])


def test(net, eval_loader, epoch=1, max_psnr_val=0, Dname='val', save_path='./'):
    net.eval()
    net_eval = net.module if isinstance(net, DDP) else net
    with torch.no_grad():
        eval_meters = AverageMeters()
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)

            residual = net_eval(inputs, args.tmat)
            outputs = inputs - residual

            eval_meters.update({
                'out_psnr': compute_psnr(outputs, labels),
                'in_psnr': compute_psnr(inputs, labels),
            })

        out_psnr = eval_meters['out_psnr']
        in_psnr = eval_meters['in_psnr']

        if out_psnr > max_psnr_val:
            max_psnr_val = out_psnr
            torch.save(net_eval.state_dict(), save_path + 'best_model.pth')

        print(f"[{Dname}] num:{len(eval_loader)} in_psnr:{in_psnr:.2f} "
              f"out_psnr:{out_psnr:.2f} best:{max_psnr_val:.2f} time:{time.time()-st:.1f}s")
        logging.info(f"[{Dname}] out_psnr:{out_psnr:.2f} best:{max_psnr_val:.2f}")

    return max_psnr_val


def get_training_dataset():
    if args.training_path_txt:
        datasets_list = []
        for txt_path in args.training_path_txt:
            ds = my_dataset_wTxt(args.training_in_path, txt_path,
                                 crop_size=args.Crop_patches,
                                 fix_sample_A=args.fix_sampleA)
            datasets_list.append(ds)
        return ConcatDataset(datasets_list)
    else:
        return my_dataset(args.training_in_path, args.training_gt_path,
                          crop_size=args.Crop_patches,
                          fix_sample_A=args.fix_sampleA)


def get_eval_data():
    eval_data = my_dataset_eval(root_in=args.eval_in_path, root_label=args.eval_gt_path,
                                transform=trans_eval, fix_sample=500)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    return eval_loader


if __name__ == '__main__':
    rank, local_rank, world_size = init_dist()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main_process():
        print(f'device: {device} (world_size={world_size})')
        print(f'auto tmat for {args.task}: {args.tmat}')

    exper_name = args.experiment_name
    SAVE_PATH = args.unified_path + exper_name + '/'

    if is_main_process():
        os.makedirs(args.writer_dir, exist_ok=True)
        os.makedirs(SAVE_PATH, exist_ok=True)
        writer = SummaryWriter(args.writer_dir + exper_name)
        logging.basicConfig(filename=SAVE_PATH + exper_name + '.log', level=logging.INFO)
        for k in args.__dict__:
            logging.info(k + ": " + str(args.__dict__[k]))
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    net = EEDTPRestorationNet(
        img_channel=args.img_channel,
        width=args.base_channel,
        middle_blk_num=args.num_res,
        enc_blk_nums=args.enc_blks,
        dec_blk_nums=args.dec_blks,
    )

    if args.load_pre_model and args.pre_model:
        net.load_state_dict(torch.load(args.pre_model, map_location='cpu'), strict=True)
        if is_main_process():
            print('loaded pre-trained model from', args.pre_model)
            logging.info('loaded pre-trained model from ' + args.pre_model)

    net.to(device)

    if is_main_process():
        print(f'#parameters: {sum(p.numel() for p in net.parameters()):,}')

    schedule = NoiseSchedule(max_sigma=10, T=args.diffusion_T, schedule='linear', eps=0.005, device=device)

    # parameter regularization setup (before DDP wrap)
    param_reg = ParameterRegularizer(net)
    if args.load_pre_model:
        if is_main_process():
            print('computing importance weights ...')
        imp_dataset = get_training_dataset()
        imp_sampler = DistributedSampler(imp_dataset) if world_size > 1 else None
        importance_loader = DataLoader(
            dataset=imp_dataset, batch_size=args.BATCH_SIZE,
            num_workers=8, shuffle=(imp_sampler is None),
            sampler=imp_sampler, drop_last=True)
        param_reg.compute_importance(
            net, importance_loader, schedule, device,
            num_batches=args.importance_batches)
        del importance_loader, imp_dataset, imp_sampler
        gc.collect()
        if is_main_process():
            print('importance weights computed')

    # wrap with DDP after importance computation
    if world_size > 1:
        net = DDP(net, device_ids=[local_rank])
    net_without_ddp = net.module if world_size > 1 else net

    train_dataset = get_training_dataset()
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.BATCH_SIZE,
        num_workers=8, shuffle=(train_sampler is None),
        sampler=train_sampler, drop_last=True)

    if is_main_process():
        eval_loader = get_eval_data()
        print(f'len(train_loader): {len(train_loader)}')

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_iters, eta_min=1e-7)

    running_results = {'iter_nums': 0, 'max_psnr_val': 0}
    train_meters = AverageMeters()
    criterion_fft = fftLoss().to(device)

    epoch = 0
    while running_results['iter_nums'] < args.total_iters:
        epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        st = time.time()

        for i, train_data in enumerate(train_loader, 0):
            data_in, label, img_name = train_data

            running_results['iter_nums'] += 1
            net.train()

            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)

            # restoration: predict residual LQ - GT
            residual = net(inputs, args.tmat)
            outputs = inputs - residual
            loss_content = F.l1_loss(outputs, labels)
            loss_fft = criterion_fft(outputs, labels)
            total_loss = loss_content + args.lambda_fft * loss_fft

            loss_gen = torch.tensor(0.)
            loss_reg = torch.tensor(0.)
            loss_orthog = torch.tensor(0.)

            if random.random() < args.gen_prob:
                # denoising loss on GT to preserve generative prior
                loss_gen = compute_denoising_loss(net, labels, schedule)

                loss_orthog = GradientOrthogonalLoss.compute(
                    net, loss_gen, loss_content)

                loss_reg = param_reg.loss(net_without_ddp, lambda_reg=args.lambda_reg)

                total_loss = total_loss + loss_reg + loss_orthog

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            in_psnr = compute_psnr(inputs, labels)
            out_psnr = compute_psnr(outputs, labels)

            train_meters.update({
                'loss': total_loss.item(),
                'loss_content': loss_content.item(),
                'loss_fft': loss_fft.item(),
                'loss_gen': loss_gen.item() if isinstance(loss_gen, torch.Tensor) else 0.,
                'loss_reg': loss_reg.item() if isinstance(loss_reg, torch.Tensor) else 0.,
                'loss_orthog': loss_orthog.item() if isinstance(loss_orthog, torch.Tensor) else 0.,
                'in_psnr': in_psnr,
                'out_psnr': out_psnr,
            })

            if is_main_process() and running_results['iter_nums'] % args.print_frequency == 0:
                writer.add_scalars(exper_name + '/training', {
                    'out_PSNR': train_meters['out_psnr'],
                    'loss': train_meters['loss'],
                }, running_results['iter_nums'])

                print("iter:%d lr:%.7f loss:%.5f(l1:%.4f,fft:%.4f,gen:%.4f,reg:%.4f,ort:%.4f) "
                      "in:%.2f out:%.2f t:%.1f" % (
                    running_results['iter_nums'],
                    optimizer.param_groups[0]["lr"],
                    train_meters['loss'], loss_content.item(),
                    loss_fft.item(),
                    loss_gen.item() if isinstance(loss_gen, torch.Tensor) else 0.,
                    loss_reg.item() if isinstance(loss_reg, torch.Tensor) else 0.,
                    loss_orthog.item() if isinstance(loss_orthog, torch.Tensor) else 0.,
                    in_psnr, out_psnr, time.time() - st))
                logging.info("iter:%d loss:%.5f in:%.2f out:%.2f" % (
                    running_results['iter_nums'], train_meters['loss'], in_psnr, out_psnr))
                st = time.time()

            if is_main_process() and running_results['iter_nums'] % args.val_frequency == 0:
                running_results['max_psnr_val'] = test(
                    net=net, eval_loader=eval_loader,
                    epoch=running_results['iter_nums'],
                    max_psnr_val=running_results['max_psnr_val'],
                    Dname='val', save_path=SAVE_PATH)

            if is_main_process() and running_results['iter_nums'] % args.save_frequency == 0:
                torch.save(net_without_ddp.state_dict(), SAVE_PATH + 'latest_model.pth')

            if running_results['iter_nums'] >= args.total_iters:
                break

    if is_main_process():
        torch.save(net_without_ddp.state_dict(), SAVE_PATH + 'latest_model.pth')
        print('Training finished.')
        writer.close()

    cleanup()
