import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
import os
import numpy as np
from torch.utils.data import Dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from PIL import Image
from MRIP import MRIPF
import argparse
import random
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel._functions import Scatter

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except Exception:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class BalancedDataParallel(DataParallel):

    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids)
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)


class CustomDataset(Dataset):
    def __init__(self, input_rgb_dir):
        self.input_rgb_files = sorted(os.listdir(input_rgb_dir))
        self.input_rgb_dir = input_rgb_dir
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        input_rgb_path = os.path.join(self.input_rgb_dir, self.input_rgb_files[index])
        rgb_dirname = os.path.dirname(input_rgb_path) + '/'
        ir_dirname = rgb_dirname.replace('reflect/', 'ir/')
        ir_basename = os.path.basename(input_rgb_path)
        ir_filename, ir_extension = os.path.splitext(ir_basename)
        input_ir_path = os.path.join(ir_dirname, ir_filename.split('_')[1] + '.png')
        input_gt_path = input_ir_path.replace('ir', 'gt')

        input_rgb = Image.open(input_rgb_path)
        input_ir = Image.open(input_ir_path)
        input_gt = Image.open(input_gt_path)



        if self.transform is not None:
            input_rgb = self.transform(input_rgb)
            input_ir = self.transform(input_ir)
            input_gt = self.transform(input_gt)

        return input_rgb, input_ir, input_gt

    def __len__(self):
        return len(self.input_rgb_files)

def psnr(input, target, max_val=1.):
    mse = F.mse_loss(input, target)
    return 10 * torch.log10(max_val**2 / mse)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1
        self.balance = 1.1

    def forward(self, inputs, targets):
        n, c, h, w = inputs.size()

        input_flat=inputs.view(-1)
        target_flat=targets.view(-1)

        intersecion=input_flat * target_flat
        unionsection=input_flat.pow(2).sum() + target_flat.pow(2).sum() + self.smooth
        loss=unionsection/(2 * intersecion.sum() + self.smooth)
        loss=loss.sum()

        return loss

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='MIRR/reflect_train/')
    parser.add_argument('--test', type=str, default='MIRR/reflect_test/')
    parser.add_argument('--savedir', type=str, default='savedir/')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--warmup_epoch', type=int, default=30)
    parser.add_argument('--fusion_epoch', type=int, default=70)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg()

    random.seed(114514)
    np.random.seed(114514)
    torch.manual_seed(114514)
    torch.cuda.manual_seed_all(114514)

    train_dataset = CustomDataset(args.train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    if torch.cuda.is_available():
        model = MRIPF().cuda()
    if torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(MRIPF().cuda())
        model = BalancedDataParallel(4, MRIPF(), dim=0).cuda()

    criterion = torch.nn.MSELoss()
    criterion_recon = CharbonnierLoss()
    criterion_grad = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)

    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            args.epoch - args.fusion_epoch + args.warmup_epoch,
                                                            eta_min=args.lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.fusion_epoch,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()


    max_psnr = 0

    for epoch in range(args.epoch):

        for param in model.parameters():
            param.grad = None

        model.train()
        running_loss = 0.0
        for (rgb, nir, input_gt) in tqdm(train_loader):
            rgb, nir, input_gt = rgb.cuda(), nir.cuda(), input_gt.cuda()

            optimizer.zero_grad()

            outputs = model(rgb, nir)
            rgb_recons, rgb_out1, nir_recons = outputs

            loss1 = criterion(rgb_recons, input_gt)
            # loss2 = criterion(rgb_recons, rgb)
            # loss3 = criterion(nir_recons, nir)

            loss_rgb = criterion_recon(torch.clamp(rgb_recons, 0, 1), input_gt)
            loss_nir = criterion_recon(torch.clamp(rgb_out1, 0, 1), nir)

            loss = loss_rgb + loss_nir

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        grid_img = make_grid(rgb_recons, nrow=8)
        save_image(grid_img, 'train_output/{}.png'.format(epoch))
        psnr1 = psnr(rgb_recons, input_gt)
        print(f'Epoch [{epoch + 1}/{args.epoch}], Loss: {running_loss / len(train_loader)}, PSNR: {psnr1}')
        if psnr1 >= max_psnr:
            torch.save(model.state_dict(), "trained_model/loss{}_psnr{}.pth".format(running_loss, psnr1))
            max_psnr = psnr1
    print('Finished Training')


