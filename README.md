# Solar Full-Disk Hyperspectral Patch-wise Convolutional Autoencoder (CAE)

## 项目简介
本项目针对高光谱太阳全日面图像，利用卷积自编码器（CAE）进行谱维降维，将原始118通道压缩到约16通道，同时保持空间分辨率不变，便于后续太阳耀斑等目标检测任务。

## 主要功能
- **高光谱FITS数据处理**：支持大尺寸FITS文件的高效分块读取与归一化。
- **Patch数据集构建**：自动生成patch索引，支持随机采样、方差/非零过滤。
- **卷积自编码器训练**：仅压缩通道维度，空间结构保持，支持残差、空洞卷积。
- **混合精度与断点恢复**：节省显存，支持AMP和断点续训。
- **模型推理与编码**：训练后可批量编码整幅图像，输出降维结果。
- **模块化结构**：各功能模块独立，易于维护和扩展。

## 目录结构
```
SolarCAE/
├── run.py                # 项目入口脚本
├── config.py             # 参数解析与配置
├── dataset/
│   └── solar_patch.py    # Patch数据集定义
├── encode/
│   └── encode_util.py    # 整图编码工具
├── loss/
│   └── loss.py           # 损失函数模块
├── model/
│   └── cae.py            # 卷积自编码器模型
├── train/
│   ├── checkpoint.py     # 模型保存与加载
│   └── train_loop.py     # 训练/验证循环
├── utils/
│   ├── io.py             # FITS读写与归一化
│   ├── patch.py          # patch网格生成
│   └── seed.py           # 随机种子与设备选择
└── README.md             # 项目说明
```

## 快速开始
### 1. 安装依赖
```bash
pip install numpy torch torchvision astropy tqdm pyyaml
```

### 2. 训练模型
```bash
python run.py \
  --data-root raw \
  --fits-glob "*HA.fits" \
  --patch-size 256 \
  --stride 256 \
  --latent-ch 16 \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-3 \
  --save-dir runs/cae16
```
可选参数：`--ssim-w 0.2` 启用SSIM损失，`--augment` 启用数据增强。

### 3. 编码整幅图像
```bash
python run.py \
  --data-root raw \
  --fits-glob "*HA.fits" \
  --patch-size 256 \
  --stride 256 \
  --latent-ch 16 \
  --checkpoint runs/cae16/best.pt \
  --encode-only \
  --encode-out encoded16
```

## 参数说明
- `--data-root`：FITS数据文件夹路径
- `--fits-glob`：FITS文件匹配模式
- `--patch-size`：patch尺寸（默认256）
- `--stride`：patch步长（默认256）
- `--latent-ch`：降维后通道数（如16）
- `--epochs`：训练轮数
- `--batch-size`：批次大小
- `--lr`：学习率
- `--save-dir`：模型保存目录
- `--checkpoint`：推理/断点恢复模型路径
- `--encode-only`：仅编码，不训练
- `--encode-out`：编码输出目录
- 其他参数详见`config.py`和`run.py`注释

## 依赖环境
- Python 3.8+
- numpy, torch, torchvision
- astropy, tqdm, pyyaml

## 典型流程
1. 数据准备：将FITS文件放入`raw/`目录。
2. 训练CAE模型，保存最佳模型到`runs/cae16/best.pt`。
3. 使用最佳模型批量编码整幅图像，输出降维结果到`encoded16/`。

## 参考与致谢
- 本项目参考了太阳物理领域高光谱数据处理与深度学习降维相关文献。
- 感谢开源社区的相关工具和库。

---
如有问题或建议，欢迎提交Issue或联系作者。
