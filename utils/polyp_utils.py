"""PolypCurriSeg 的难度先验与 EPSB 频率门控。

本文件中的函数仅在训练时使用，不会给 Network 增加推理参数，默认输入尺寸
适合 8--12 GB 显存。
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def _read_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _read_mask(path: str) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return (mask > 127).astype(np.uint8)


def polyp_difficulty_prior(image_path: str, gt_path: str) -> Dict[str, float]:
    """计算确定性的息肉医学域难度先验。

    低对比度和边界模糊值越大，表示息肉越难与背景分离；边界复杂度来自 GT
    边界像素比例。反光高亮被显式作为干扰项扣分，避免仅因反光就把样本提升
    为有价值难例。
    """

    image = _read_rgb(image_path)
    mask = _read_mask(gt_path)
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    fg = mask > 0
    kernel = np.ones((5, 5), np.uint8)
    dil = cv2.dilate(mask, kernel, iterations=1)
    ero = cv2.erode(mask, kernel, iterations=1)
    boundary = (dil - ero) > 0
    boundary_ratio = float(boundary.mean())

    # 目标周围的环形区域比整幅图更稳定。
    ring = (dil > 0) & (~fg)
    if fg.any() and ring.any():
        contrast = abs(float(gray[fg].mean()) - float(gray[ring].mean()))
    elif fg.any():
        contrast = abs(float(gray[fg].mean()) - float(gray.mean()))
    else:
        contrast = 0.0
    low_contrast = float(np.clip(1.0 - contrast / 0.35, 0.0, 1.0))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    boundary_grad = float(grad[boundary].mean()) if boundary.any() else 0.0
    global_grad = float(np.percentile(grad, 75)) + 1e-6
    sharpness = float(np.clip(boundary_grad / (global_grad * 1.5), 0.0, 1.0))
    blurred_boundary = 1.0 - sharpness

    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    specular = ((hsv[..., 2] > 235) & (hsv[..., 1] < 45)).astype(np.float32)
    specular_ratio = float(np.clip(specular.mean() * 8.0, 0.0, 1.0))

    # 反光惩罚是有意设计：反光属于噪声难度，不应在课程选择中被过采样。
    prior = (
        0.40 * low_contrast
        + 0.30 * blurred_boundary
        + 0.30 * float(np.clip(boundary_ratio * 12.0, 0.0, 1.0))
        - 0.45 * specular_ratio
    )
    return {
        "prior": float(np.clip(prior, 0.0, 1.0)),
        "low_contrast": low_contrast,
        "blurred_boundary": float(blurred_boundary),
        "boundary_complexity": float(np.clip(boundary_ratio * 12.0, 0.0, 1.0)),
        "specular_ratio": specular_ratio,
    }


def _binary_boundary(mask: torch.Tensor, width: int = 2) -> torch.Tensor:
    """用池化近似形态学边界，便于放在张量计算中。"""
    k = max(1, int(width) * 2 + 1)
    dil = F.max_pool2d(mask, k, stride=1, padding=k // 2)
    ero = -F.max_pool2d(-mask, k, stride=1, padding=k // 2)
    return (dil - ero).clamp(0.0, 1.0)


def boundary_protected_frequency_gate(
    images: torch.Tensor,
    gts: torch.Tensor,
    pred_logits: Optional[torch.Tensor] = None,
    cutoff_ratio: float = 0.18,
    suppress_strength: float = 0.75,
    boundary_width: int = 2,
) -> torch.Tensor:
    """EPSB：边界带内保留高频，边界带外抑制高频。

    单个径向 FFT 掩码保证操作轻量；边界带外只保留部分高频，边界带内完全
    保留高频细节。pred_logits 可选，微调时可传入一次无梯度预测作为边界。
    """
    if images.ndim != 4:
        raise ValueError("images must have shape [B,C,H,W]")
    _, _, h, w = images.shape
    target = gts.float().clamp(0.0, 1.0)
    if target.shape[-2:] != (h, w):
        target = F.interpolate(target, size=(h, w), mode="nearest")
    band = _binary_boundary(target, boundary_width)
    if pred_logits is not None:
        pred = (torch.sigmoid(pred_logits) > 0.5).float()
        if pred.shape[-2:] != (h, w):
            pred = F.interpolate(pred, size=(h, w), mode="nearest")
        band = torch.maximum(band, _binary_boundary(pred, boundary_width))
    # 将细轮廓扩展为稳定的保护带。
    band = F.max_pool2d(band, kernel_size=2 * boundary_width + 1, stride=1, padding=boundary_width)

    fy = torch.fft.fftfreq(h, device=images.device).view(h, 1)
    fx = torch.fft.rfftfreq(w, device=images.device).view(1, -1)
    radius = torch.sqrt(fx * fx + fy * fy)
    cutoff = max(float(cutoff_ratio), 1e-3) * 0.5
    low_mask = (radius <= cutoff).to(images.dtype)[None, None]
    spectrum = torch.fft.rfft2(images, norm="ortho")
    low = torch.fft.irfft2(spectrum * low_mask, s=(h, w), norm="ortho").real
    high = images - low
    retain = 1.0 - float(suppress_strength) * (1.0 - band)
    return low + high * retain


def get_dataset_prior(dataset, idx: int) -> float:
    priors = getattr(dataset, "difficulty_priors", None)
    if priors is None:
        return 0.0
    return float(priors.get(int(idx), {}).get("prior", 0.0))
