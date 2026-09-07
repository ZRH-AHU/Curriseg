# PolypCurriSeg

这是一个以息肉图像分割为主线的 CurriSeg 改造版。方法保持轻量主干和两阶段训练：

1. **阶段 1：Robust Curriculum Selection（RCS）**。用样本损失/IoU 的时间统计选择样本，并加入息肉医学域先验：低对比度、边界模糊、边界复杂度。反光高亮作为干扰项扣分，避免把反光噪声误当成有价值难例。
2. **阶段 2：Anti-curriculum + EPSB**。在模型最难子集上微调。EPSB 使用 FFT 低通/高频分解，在预测或标注边界带内保留高频，在边界带外抑制纹理高频，避免息肉轮廓变糊。EPSB 只在训练时使用，不增加推理参数。

代码不声称临床有效性，建议把实验主张限定为跨数据集/跨设备泛化、边界质量和小显存可复现性。

## 数据目录

将任意公开数据集整理成以下结构；`Edge/` 可以省略，代码会从 GT 自动生成边界监督。

```text
dataset_root/
  Imgs/                 # RGB 图像，jpg/png/jpeg/bmp/tif 均可
  GT/                   # 二值掩码，与图像文件名 stem 对齐
  Edge/                 # 可选；已有边界标签时放在这里
```

图像和 GT 的文件名 stem 应一致，例如 `Imgs/0001.jpg` 对应 `GT/0001.png`。

## 安装

```bash
conda create -n polypcurriseg python=3.10
conda activate polypcurriseg
pip install -r requirements.txt
```

## 训练

阶段 1（RCS）：

```bash
python Train.py \
  --train_root D:/data/Kvasir-SEG/train \
  --val_root D:/data/Kvasir-SEG/val \
  --save_path runs/kvasir_rcs \
  --trainsize 384 --batchsize 8 --num_workers 4
```

显存只有 8--12 GB 时，优先把 `--batchsize` 调到 2--8。训练期间会在 `save_path` 保存难度映射和 TSSW 时间统计。

阶段 2（hard-subset + EPSB）：

```bash
python anti_curri_stage.py \
  --train_root D:/data/Kvasir-SEG/train \
  --val_root D:/data/Kvasir-SEG/val \
  --save_path runs/kvasir_epsb \
  --load runs/kvasir_rcs/Net_epoch_best.pth \
  --trainsize 384 --batchsize 8 --hard_ratio 0.2 \
  --epsb_prob 0.7 --epsb_cutoff 0.18 --epsb_suppress 0.75
```

消融：`--no_epsb` 关闭 EPSB；`--epsb_no_pred_boundary` 只用 GT 边界；将 `--hard_ratio 1.0` 可近似关闭 hard-subset；阶段 1/2 的 `--prior_weight 0` 可得到不使用医学域先验的基线。

## 测试

```bash
python Test.py \
  --pth_path runs/kvasir_epsb/Net_epoch_best.pth \
  --test_image_root D:/data/Kvasir-SEG/test/Imgs \
  --test_gt_root D:/data/Kvasir-SEG/test/GT \
  --save_path runs/kvasir_epsb/predictions
```

`Test.py` 输出每张预测图和 MAE。论文实验建议额外计算 Dice、IoU、S-measure、F-measure、Hausdorff 距离/ASSD，以及边界 F-score 或 trimap IoU。

## 建议的公开息肉数据集

- **[Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)**：1000 张带像素级掩码的胃肠息肉图像，适合主训练集或常规验证。
- **[CVC-ClinicDB（CVC-612）](https://polyp.grand-challenge.org/CVCClinicDB/)**：612 张、来自多段结肠镜序列，适合跨中心/跨设备测试。
- **[ETIS-LaribPolypDB](https://polyp.grand-challenge.org/EtisLarib/)**：独立实验室采集的小规模测试集，常用于检验跨数据集泛化。
- **[CVC-ColonDB](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/)**：经典独立测试集，适合与 CVC-ClinicDB 组合做域外评估。
- **[CVC-300 / EndoScene](http://adas.cvc.uab.es/endoscene)**：EndoScene 中的 CVC-300 子集，常用于外部测试。
- **[SUN-SEG](https://github.com/Gewtial/SUN-SEG)**：大规模结肠镜视频息肉分割基准，包含 seen/unseen、easy/hard 划分，适合视频帧和跨域泛化分析。
- **[PolypGen](https://github.com/DebeshJha/PolypGen)**：多中心、多序列数据，适合研究中心间域偏移；使用前请按官方许可和标注协议整理分割子集。

建议至少采用“一个数据集训练、其余数据集完全不参与训练”的协议，例如 Kvasir-SEG 训练，CVC-ClinicDB、CVC-ColonDB、ETIS、CVC-300、SUN-SEG 做外部测试；不要把不同数据集的相邻视频帧随机混到训练和测试中。

## 代码对应关系

- `Train.py`：阶段 1，时间统计 + 医学域难度先验的课程选择。
- `anti_curri_stage.py`：阶段 2，hard-subset 和 EPSB 频域微调。
- `utils/polyp_utils.py`：息肉难度先验和边界保护频率门控。
- `utils/data_val.py`：图像/GT 配对、增强和自动边界监督。
- `Test.py`：单一图像目录/GT 目录的批量推理。
