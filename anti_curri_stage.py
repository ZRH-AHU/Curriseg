#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PolypCurriSeg 第二阶段：难例子集反课程微调与 EPSB。"""

import os
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

from lib.Network import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, get_coef, cal_ual
from utils.polyp_utils import boundary_protected_frequency_gate, get_dataset_prior


# -----------------------------
# Globals
# -----------------------------
device_ids = [0]
best_mae = 1.0
best_epoch = 0
step = 0


# -----------------------------
# Losses (SAME AS YOUR CURRENT)
# -----------------------------
def dice_loss(predict, target, smooth=1.0, p=2.0):
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def structure_loss(pred_logits, mask):
    """Original structure loss (NO extra weighting)"""
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )

    bce = F.binary_cross_entropy_with_logits(pred_logits, mask, reduction='none')
    wbce = (weit * bce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-6)

    pred = torch.sigmoid(pred_logits)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1.0) / (union - inter + 1.0)

    return wbce + wiou


# -----------------------------
# Difficulty = 1 - softIoU
# -----------------------------
def batch_soft_iou_from_logits(logits, gts, eps=1e-6):
    p = torch.sigmoid(logits)
    g = gts.float()
    inter = (p * g).sum(dim=(1, 2, 3))
    union = (p + g - p * g).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return iou


@torch.no_grad()
def compute_difficulty_by_model(model, full_loader, device, final_idx=4, prior_weight=None):
    """计算 PolypCurriSeg 难例分数：模型误差加确定性的医学域先验。

    先验包含低对比度、边界模糊和边界复杂度，反光干扰已在先验中降权。
    返回值为样本索引到难度的映射。
    """
    model.eval()
    d_map = {}
    if prior_weight is None:
        prior_weight = float(getattr(globals().get('opt', None), 'prior_weight', 0.35))
    prior_weight = float(np.clip(prior_weight, 0.0, 1.0))
    dataset = full_loader.dataset
    old_augment = getattr(dataset, 'augment', None)
    if old_augment is not None:
        dataset.augment = False
    try:
        for images, gts, edges, idxs in full_loader:
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)

            preds = model(images)
            iou = batch_soft_iou_from_logits(preds[final_idx], gts)
            d = 1.0 - iou

            if torch.is_tensor(idxs):
                idxs = idxs.detach().cpu().tolist()

            for j, idx in enumerate(idxs):
                prior = get_dataset_prior(dataset, int(idx))
                d_map[int(idx)] = float((1.0 - prior_weight) * d[j].item() + prior_weight * prior)
    finally:
        if old_augment is not None:
            dataset.augment = old_augment

    return d_map


def summarize_difficulty(d_map, tag="difficulty"):
    ds = np.array(list(d_map.values()), dtype=np.float32)
    if ds.size == 0:
        print(f"[DIFF STAT] {tag}: empty")
        return

    print(
        f"[DIFF STAT] {tag}: N={len(ds)} "
        f"min={ds.min():.6f} p10={np.quantile(ds, 0.1):.6f} p50={np.quantile(ds, 0.5):.6f} "
        f"p90={np.quantile(ds, 0.9):.6f} max={ds.max():.6f} mean={ds.mean():.6f} std={ds.std():.6f}"
    )


# -----------------------------
# Robust idx -> dataset_index mapping
# -----------------------------
def build_idx_to_dataset_index(dataset, max_print=5):
    """
    dataset[i] returns (img, gt, edge, idx)
    We build mapping:
      returned idx  -> actual dataset index i
    This makes Subset safe even if idx != i
    """
    idx2i = {}
    dup = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        idx = int(sample[-1])
        if idx in idx2i:
            dup += 1
            if dup <= max_print:
                print(f"[IDX MAP] duplicate idx found: idx={idx} already mapped to {idx2i[idx]}, new={i}")
        else:
            idx2i[idx] = i

    print(f"[IDX MAP] built: {len(idx2i)}/{len(dataset)} (dup={dup})")
    return idx2i


def select_hardest_subset(d_map, hard_ratio=0.2):
    """
    Pick hardest top hard_ratio by difficulty descending
    returns: list of idx (the returned idx from dataset)
    """
    pairs = [(k, float(v)) for k, v in d_map.items()]
    pairs.sort(key=lambda x: x[1], reverse=True)

    k = max(1, int(np.ceil(len(pairs) * hard_ratio)))
    hard_idxs = [p[0] for p in pairs[:k]]
    return hard_idxs


# -----------------------------
# Validation (SAME STYLE)
# -----------------------------
def val(test_loader, model, epoch, save_path, writer):
    global best_mae, best_epoch

    model.eval()
    with torch.no_grad():
        mae_sum = 0.0

        for _ in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            image = image.cuda(device=device_ids[0], non_blocking=True)

            result = model(image)
            res = F.interpolate(result[4], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            mae_sum += np.sum(np.abs(res - gt)) / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / float(test_loader.size)
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)

        print(f'[Val] Epoch: {epoch}, MAE: {mae:.6f}, bestMAE: {best_mae:.6f}, bestEpoch: {best_epoch}')
        logging.info(f'[Val Info]:Epoch:{epoch} MAE:{mae} bestEpoch:{best_epoch} bestMAE:{best_mae}')

        if epoch == 1:
            best_mae = mae
            best_epoch = 1
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_best.pth'))
                print(f'[Val] Save best state_dict! Best epoch: {epoch}')
                logging.info(f'[Val] Save best state_dict! Best epoch: {epoch}')


# -----------------------------
# Anti-curriculum training loop
# -----------------------------
def train_one_epoch_anti_curri(
    hard_loader,
    model,
    optimizer,
    epoch,
    writer,
    use_epsb=True,
    epsb_prob=0.7,
    epsb_cutoff=0.18,
    epsb_suppress=0.75,
    epsb_boundary_width=2,
    epsb_use_pred_boundary=True,
):
    global step

    model.train()
    device = next(model.parameters()).device
    total_step = max(len(hard_loader), 1)

    loss_all = 0.0
    epoch_step = 0

    for it, (images, gts, edges, idxs) in enumerate(hard_loader, start=1):
        optimizer.zero_grad(set_to_none=True)

        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)
        edges = edges.to(device, non_blocking=True)

        # EPSB：预测边界只用于构造训练期 FFT 门控，推理阶段不增加参数。
        if use_epsb and (np.random.rand() < epsb_prob):
            pred_preview = None
            if epsb_use_pred_boundary:
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    pred_preview = model(images)[4]
                if was_training:
                    model.train()
            images = boundary_protected_frequency_gate(
                images, gts, pred_logits=pred_preview,
                cutoff_ratio=epsb_cutoff,
                suppress_strength=epsb_suppress,
                boundary_width=epsb_boundary_width,
            )

        preds = model(images)

        # ---- UAL ----
        ual_coef = get_coef(iter_percentage=it / float(total_step), method='cos')
        ual_loss = cal_ual(seg_logits=preds[4], seg_gts=gts)
        ual_loss = ual_loss * ual_coef

        # ---- loss (same as your current) ----
        loss_init = (
            structure_loss(preds[0], gts).mean() * 0.0625 +
            structure_loss(preds[1], gts).mean() * 0.125 +
            structure_loss(preds[2], gts).mean() * 0.25 +
            structure_loss(preds[3], gts).mean() * 0.5
        )
        loss_final = structure_loss(preds[4], gts).mean()
        loss_edge = (
            dice_loss(preds[5], edges) * 0.0625 +
            dice_loss(preds[6], edges) * 0.125 +
            dice_loss(preds[7], edges) * 0.25 +
            dice_loss(preds[8], edges) * 0.5
        )

        loss = loss_init + loss_final + loss_edge + 2.0 * ual_loss

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        epoch_step += 1
        loss_all += float(loss.item())
        step += 1

        if it % 20 == 0 or it == 1 or it == len(hard_loader):
            print(f'{datetime.now()} [Anti-curri] Epoch [{epoch}/{opt.epoch}] Step [{it}/{len(hard_loader)}] '
                  f'Loss: {loss.item():.4f}')
            writer.add_scalars('Loss_Statistics', {
                'Loss_total': loss.item(),
                'Loss_init': float(loss_init.item()),
                'Loss_final': float(loss_final.item()),
                'Loss_edge': float(loss_edge.item()),
            }, global_step=step)

    loss_all /= max(epoch_step, 1)
    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
    logging.info(f'[Anti-curri Train] Epoch [{epoch}/{opt.epoch}] Loss_AVG: {loss_all:.6f}')
    return loss_all


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # 训练参数
    parser.add_argument('--epoch', type=int, default=100, help='anti-curri total epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='finetune learning rate')
    parser.add_argument('--batchsize', type=int, default=36, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training image size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

    # 路径参数
    parser.add_argument('--train_root', type=str, default='',
                        help='息肉训练集目录（包含 Imgs/ 和 GT/，Edge/ 可省略）')
    parser.add_argument('--val_root', type=str, default='',
                        help='息肉验证集目录（包含 Imgs/ 和 GT/）')
    parser.add_argument('--save_path', type=str, default='',
                        help='path to save model and log')
    parser.add_argument('--load', type=str, default='',
                        help='path to load ckpt (your bestMAE model, e.g., Net_epoch_best.pth)')

    # 反课程参数
    parser.add_argument('--hard_ratio', type=float, default=0.2,
                        help='hardest ratio used for anti-curri')
    parser.add_argument('--recompute_diff_every', type=int, default=0,
                        help='recompute difficulty every N epochs')
    parser.add_argument('--prior_weight', type=float, default=0.35,
                        help='息肉医学域先验权重；设为 0 可做无先验基线')
    parser.add_argument('--num_workers', type=int, default=16, help='dataloader workers')

    # EPSB (Boundary-protected frequency fine-tuning)
    parser.add_argument('--no_epsb', action='store_true', help='消融时关闭 EPSB')
    parser.add_argument('--epsb_prob', type=float, default=0.7, help='每个 batch 使用 EPSB 的概率')
    parser.add_argument('--epsb_cutoff', type=float, default=0.18, help='径向低通截止比例')
    parser.add_argument('--epsb_suppress', type=float, default=0.75, help='边界外高频抑制强度')
    parser.add_argument('--epsb_boundary_width', type=int, default=2, help='边界保护带半径（像素）')
    parser.add_argument('--epsb_no_pred_boundary', action='store_true', help='只使用 GT 边界的低成本消融')

    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    cudnn.benchmark = True

    save_path = opt.save_path
    os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_path, 'anti_curri.log'),
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )

    logging.info('PolypCurriSeg 阶段二：hard-subset + EPSB')
    logging.info(
        f'Config: epoch={opt.epoch} lr={opt.lr} batchsize={opt.batchsize} trainsize={opt.trainsize} '
        f'hard_ratio={opt.hard_ratio} prior_weight={opt.prior_weight} '
        f'use_epsb={not opt.no_epsb} load={opt.load}'
    )

    # Build model
    model = Network(channels=192).cuda(device=device_ids[0])

    # Load checkpoint
    ckpt = torch.load(opt.load, map_location='cuda')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    new_state = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(new_state, strict=False)
    print(f"[Load] Loaded ckpt: {opt.load}")
    logging.info(f"[Load] Loaded ckpt: {opt.load}")

    # Optimizer (fine-tune)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=1e-4
    )

    # Data
    print('[Data] Loading...')
    train_loader = get_loader(
        image_root=os.path.join(opt.train_root, 'Imgs/'),
        gt_root=os.path.join(opt.train_root, 'GT/'),
        edge_root=os.path.join(opt.train_root, 'Edge/'),
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        num_workers=opt.num_workers
    )
    val_loader = test_dataset(
        image_root=os.path.join(opt.val_root, 'Imgs/'),
        gt_root=os.path.join(opt.val_root, 'GT/'),
        testsize=opt.trainsize
    )

    writer = SummaryWriter(os.path.join(save_path, 'summary'))

    # Full loader for difficulty compute
    full_loader = DataLoader(
        train_loader.dataset,
        batch_size=opt.batchsize,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Build idx->dataset index mapping for safe Subset
    idx2i = build_idx_to_dataset_index(train_loader.dataset)

    # Compute difficulty once at start
    print("[Anti-curri] Computing difficulty (start)...")
    d_map = compute_difficulty_by_model(model, full_loader, device='cuda', final_idx=4)
    summarize_difficulty(d_map, tag="start")

    hard_idxs = select_hardest_subset(d_map, hard_ratio=opt.hard_ratio)

    # Convert returned idx -> dataset index for Subset
    hard_dataset_indices = []
    miss = 0
    for idx in hard_idxs:
        if idx in idx2i:
            hard_dataset_indices.append(idx2i[idx])
        else:
            miss += 1
    print(f"[Anti-curri] Hard subset: idxN={len(hard_idxs)} -> datasetN={len(hard_dataset_indices)} (miss={miss})")
    logging.info(f"[Anti-curri] Hard subset datasetN={len(hard_dataset_indices)} miss={miss}")

    hard_loader = DataLoader(
        Subset(train_loader.dataset, hard_dataset_indices),
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Train anti-curri
    print('[Anti-curri] Start fine-tuning...')
    for epoch in range(1, opt.epoch + 1):
        cur_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate/lr', cur_lr, global_step=epoch)

        # Optionally recompute difficulty and refresh hard subset
        if opt.recompute_diff_every > 0 and (epoch > 1) and (epoch % opt.recompute_diff_every == 0):
            print(f"[Anti-curri] Recompute difficulty @epoch={epoch} ...")
            d_map = compute_difficulty_by_model(model, full_loader, device='cuda', final_idx=4)
            hard_idxs = select_hardest_subset(d_map, hard_ratio=opt.hard_ratio)

            hard_dataset_indices = []
            for idx in hard_idxs:
                if idx in idx2i:
                    hard_dataset_indices.append(idx2i[idx])

            hard_loader = DataLoader(
                Subset(train_loader.dataset, hard_dataset_indices),
                batch_size=opt.batchsize,
                shuffle=True,
                num_workers=opt.num_workers,
                pin_memory=True,
                drop_last=False
            )

            summarize_difficulty(d_map, tag=f"epoch={epoch}")
            print(f"[Anti-curri] Refreshed hard subset size={len(hard_dataset_indices)}")

        # Train one epoch
        train_one_epoch_anti_curri(
            hard_loader=hard_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            writer=writer,
            use_epsb=not opt.no_epsb,
            epsb_prob=opt.epsb_prob,
            epsb_cutoff=opt.epsb_cutoff,
            epsb_suppress=opt.epsb_suppress,
            epsb_boundary_width=opt.epsb_boundary_width,
            epsb_use_pred_boundary=not opt.epsb_no_pred_boundary,
        )

        # Validate every epoch
        val(val_loader, model, epoch, save_path, writer)

        # Save periodic checkpoint
        if epoch % 1 == 0:
            ckpt_path = os.path.join(save_path, f'Net_epoch_{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"[CKPT] Saved: {ckpt_path}")
            logging.info(f"[CKPT] Saved: {ckpt_path}")

    writer.close()
    print('[Done] Anti-curriculum fine-tuning finished.')
