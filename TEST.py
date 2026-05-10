"""Inference script for EEDTP restoration network.

Supports both single-task and multi-task unified IR with:
  - Input ensemble (4-way flip averaging)
  - Overlapped split-merge for large images
  - PSNR / SSIM evaluation

Usage:
    # single-task
    python TEST.py \
        --model_path ./ckpt/best_model.pth \
        --input_path ./data/test_input/ \
        --gt_path ./data/test_gt/ \
        --output_path ./results/ \
        --tmat 47 \
        --use_ensemble True

    # multi-task (specify task name to auto-select tmat)
    python TEST.py \
        --model_path ./ckpt/unified_model.pth \
        --input_path ./data/test_input/ \
        --task dehazing
"""
import time, torchvision, argparse, logging, sys, os
import torch, random
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
from utils.UTILS1 import compute_psnr
from utils.UTILS import AverageMeters, compute_ssim
from datasets.datasets_pairs import my_dataset_eval
from networks.eedtp_arch import EEDTPRestorationNet
from networks.image_utils import splitimage, mergeimage

sys.path.append(os.getcwd())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)


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
parser.add_argument('--model_path', type=str, default='./ckpt/best_model.pth')
parser.add_argument('--input_path', type=str, default='./data/test_input/')
parser.add_argument('--gt_path', type=str, default='./data/test_gt/')
parser.add_argument('--output_path', type=str, default='./results/')

# arch
parser.add_argument('--base_channel', type=int, default=32)
parser.add_argument('--num_res', type=int, default=6)
parser.add_argument('--img_channel', type=int, default=3)
parser.add_argument('--enc_blks', nargs='+', type=int, default=[1, 1, 1, 28])
parser.add_argument('--dec_blks', nargs='+', type=int, default=[1, 1, 1, 1])

# task
parser.add_argument('--task', type=str, default=None)
parser.add_argument('--tmat', type=int, default=None)
parser.add_argument('--diffusion_T', type=int, default=50)

# inference options
parser.add_argument('--use_ensemble', type=str2bool, default=True)
parser.add_argument('--use_split', type=str2bool, default=False)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--overlap_size', type=int, default=64)
parser.add_argument('--save_images', type=str2bool, default=True)

args = parser.parse_args()

if args.tmat is None and args.task is not None:
    args.tmat = TMAT_TABLE.get(args.task.lower(), 25)
    print(f'auto tmat for {args.task}: {args.tmat}')
elif args.tmat is None:
    args.tmat = 50
    print(f'default tmat: {args.tmat}')


def restore(net, inp, tmat):
    """Single forward: output = cond - net(cond, cond, tmat)."""
    noise_pred = net(inp, inp, tmat)
    return inp - noise_pred


def ensemble_forward(net, inp, tmat):
    """4-way flip ensemble: average predictions from original + 3 flips."""
    preds = []

    preds.append(restore(net, inp, tmat))

    inp_h = torch.flip(inp, [-1])
    preds.append(torch.flip(restore(net, inp_h, tmat), [-1]))

    inp_v = torch.flip(inp, [-2])
    preds.append(torch.flip(restore(net, inp_v, tmat), [-2]))

    inp_hv = torch.flip(inp, [-1, -2])
    preds.append(torch.flip(restore(net, inp_hv, tmat), [-1, -2]))

    return sum(preds) / len(preds)


def inference_with_split(net, inp, tmat, crop_size=256, overlap_size=64):
    """Split-merge inference for large images."""
    B, C, H, W = inp.shape
    split_data, starts = splitimage(inp, crop_size=crop_size, overlap_size=overlap_size)

    for i, patch in enumerate(split_data):
        if args.use_ensemble:
            split_data[i] = ensemble_forward(net, patch, tmat)
        else:
            split_data[i] = restore(net, patch, tmat)

    return mergeimage(split_data, starts, crop_size=crop_size,
                      resolution=(B, C, H, W), is_mean=True)


if __name__ == '__main__':
    os.makedirs(args.output_path, exist_ok=True)

    # build network
    net = EEDTPRestorationNet(
        img_channel=args.img_channel,
        width=args.base_channel,
        middle_blk_num=args.num_res,
        enc_blk_nums=args.enc_blks,
        dec_blk_nums=args.dec_blks,
    )

    # load weights
    ckpt = torch.load(args.model_path, map_location='cpu')
    missing, unexpected = net.load_state_dict(ckpt, strict=False)
    if missing:
        print(f'missing keys: {len(missing)} (MoE adapters not loaded)')
    net.to(device)
    net.eval()
    print(f'#parameters: {sum(p.numel() for p in net.parameters()):,}')
    print(f'loaded model from {args.model_path}')
    print(f'tmat: {args.tmat}, ensemble: {args.use_ensemble}, split: {args.use_split}')

    # dataset
    trans_eval = transforms.Compose([transforms.ToTensor()])
    eval_data = my_dataset_eval(
        root_in=args.input_path, root_label=args.gt_path,
        transform=trans_eval, fix_sample=10000)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    print(f'test images: {len(eval_loader)}')

    eval_meters = AverageMeters()
    st = time.time()

    with torch.no_grad():
        for index, (data_in, label, name) in enumerate(eval_loader, 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)

            if args.use_split:
                outputs = inference_with_split(
                    net, inputs, args.tmat,
                    crop_size=args.crop_size, overlap_size=args.overlap_size)
            elif args.use_ensemble:
                outputs = ensemble_forward(net, inputs, args.tmat)
            else:
                outputs = restore(net, inputs, args.tmat)

            outputs = torch.clamp(outputs, 0, 1)
            psnr = compute_psnr(outputs, labels)
            ssim = compute_ssim(outputs, labels)

            eval_meters.update({
                'psnr': psnr,
                'ssim': ssim,
                'in_psnr': compute_psnr(inputs, labels),
            })

            if args.save_images:
                img_name = name[0] if isinstance(name, (list, tuple)) else str(index)
                if not img_name.endswith('.png'):
                    img_name = img_name.split('.')[0] + '.png'
                save_path = os.path.join(args.output_path, img_name)
                torchvision.utils.save_image(outputs.cpu(), save_path)

            if (index + 1) % 50 == 0:
                print(f"  [{index+1}/{len(eval_loader)}] "
                      f"psnr:{eval_meters['psnr']:.2f} ssim:{eval_meters['ssim']:.4f}")

    print(f"\n{'='*50}")
    print(f"Results ({len(eval_loader)} images):")
    print(f"  Input  PSNR: {eval_meters['in_psnr']:.2f}")
    print(f"  Output PSNR: {eval_meters['psnr']:.2f}")
    print(f"  Output SSIM: {eval_meters['ssim']:.4f}")
    print(f"  Time: {time.time()-st:.1f}s")
    print(f"{'='*50}")
