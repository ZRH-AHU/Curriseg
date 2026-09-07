"""PolypCurriSeg 单目录推理脚本。"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.Network import Network
from utils.data_val import test_dataset


def load_weights(model, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model", state))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"加载权重：{path}，缺失 {len(missing)} 个，额外 {len(unexpected)} 个")


def main():
    parser = argparse.ArgumentParser(description="PolypCurriSeg 息肉分割推理")
    parser.add_argument("--pth_path", required=True, help="模型权重路径")
    parser.add_argument("--test_image_root", required=True, help="测试图像目录")
    parser.add_argument("--test_gt_root", required=True, help="测试 GT 目录")
    parser.add_argument("--save_path", required=True, help="预测结果目录")
    parser.add_argument("--testsize", type=int, default=384, help="网络输入边长")
    parser.add_argument("--gpu_id", default="0", help="使用的 GPU 编号；没有 GPU 时填 cpu")
    opt = parser.parse_args()

    use_cuda = opt.gpu_id.lower() != "cpu" and torch.cuda.is_available()
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device("cuda" if use_cuda else "cpu")
    os.makedirs(opt.save_path, exist_ok=True)

    model = Network(channels=192).to(device)
    load_weights(model, opt.pth_path, device)
    model.eval()
    loader = test_dataset(opt.test_image_root, opt.test_gt_root, opt.testsize)
    mae_sum = 0.0

    with torch.no_grad():
        for _ in range(loader.size):
            image, gt, name, _ = loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt = gt / (gt.max() + 1e-8)
            result = model(image.to(device))[4]
            pred = F.interpolate(result, size=gt.shape, mode="bilinear", align_corners=False)
            pred = pred.sigmoid().cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            cv2.imwrite(os.path.join(opt.save_path, name), (pred * 255).astype(np.uint8))
            mae_sum += float(np.abs(pred - gt).mean())

    print(f"样本数：{loader.size}，MAE：{mae_sum / max(loader.size, 1):.6f}")


if __name__ == "__main__":
    main()

