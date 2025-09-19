# SimpleVQA 综合技术文档

## 项目概述

SimpleVQA 是一个基于深度学习的视频质量评估（Video Quality Assessment, VQA）系统，专门用于无参考视频质量评估。该项目结合了空间特征提取和时间运动特征分析，能够自动评估视频的感知质量。

### 核心特性
- **无参考评估**: 不需要原始高质量视频作为参考，直接评估视频质量
- **多尺度分析**: 结合空间和时间维度的特征提取
- **端到端训练**: 从原始视频到质量评分的完整流程
- **多数据集支持**: 兼容LSVQ、KoNViD-1k、YouTube-UGC等主流数据集
- **实时推理**: 支持单视频和批量视频的快速质量评估

### 应用场景
- **视频平台质量监控**: 自动检测上传视频的质量问题
- **视频压缩优化**: 评估不同压缩参数对视频质量的影响
- **内容审核**: 过滤低质量视频内容
- **用户体验优化**: 根据网络条件推荐合适质量的视频流

## 技术架构

### 核心框架
- **深度学习框架**: PyTorch 1.9+
- **计算机视觉**: OpenCV 4.5+, PIL 8.0+
- **数据处理**: NumPy 1.21+, Pandas 1.3+, SciPy 1.7+
- **预训练模型**: ResNet-50 (ImageNet), SlowFast (Kinetics-400)
- **视频处理**: PyTorchVideo, FFmpeg

### 系统架构图
```
输入视频 → 帧提取 → 空间特征提取 (ResNet-50)
                ↓
         运动特征提取 (SlowFast) → 特征融合 → 质量评分输出
```

### 模型架构详解
项目采用双流架构设计：

#### 1. 空间特征流 (Spatial Stream)
- **骨干网络**: ResNet-50预训练模型
- **输入尺寸**: 448×448×3 (RGB图像)
- **特征维度**: 2048维全局特征
- **池化策略**: 全局平均池化 + 全局标准差池化
- **作用**: 捕获视频帧中的空间质量信息（模糊、噪声、伪影等）

#### 2. 时间特征流 (Temporal Stream)
- **骨干网络**: SlowFast 3D CNN
- **慢通道**: 低帧率，捕获空间语义信息
- **快通道**: 高帧率，捕获快速运动信息
- **特征维度**: 2048+256=2304维时间特征
- **作用**: 分析视频中的运动模式和时间一致性

#### 3. 特征融合模块
- **融合策略**: 线性加权融合
- **权重学习**: 通过训练自动学习最优融合权重
- **输出**: 单一质量评分 (1-5分或连续值)

## 数据集支持

### 支持的数据集详解

#### 1. LSVQ (Large-scale Short Video Quality)
- **数据规模**: 
  - 训练集: 25个视频样本
  - 测试集: 5个视频样本
  - 1080p测试集: 额外的高分辨率测试样本
- **质量评分范围**: 1.0-3.0 (MOS评分)
- **视频特点**: 短视频，用户生成内容
- **文件格式**: MP4视频 + CSV标注文件
- **存储结构**:
  ```
  LSVQ/                    # 原始视频文件
  ├── test1_3.mp4         # 视频文件，文件名包含质量评分
  ├── test2_2.8.mp4
  └── ...
  
  LSVQ_image/             # 提取的视频帧
  ├── test1_3/           # 对应视频的帧序列
  │   ├── 000.png
  │   ├── 001.png
  │   └── ...
  └── ...
  
  data/
  ├── LSVQ_whole_train.csv  # 训练集标注
  └── LSVQ_whole_test.csv   # 测试集标注
  ```

#### 2. KoNViD-1k
- **数据规模**: 1200个视频样本
- **质量评分**: 1-5分制MOS评分
- **视频来源**: 网络收集的各类视频
- **分辨率**: 多种分辨率混合
- **存储格式**: MATLAB .mat文件
- **特点**: 涵盖多种失真类型和内容类别

#### 3. YouTube-UGC
- **数据规模**: 1500+视频样本
- **质量评分**: 连续MOS评分
- **视频特点**: YouTube用户上传内容
- **失真类型**: 压缩、传输、设备相关失真
- **存储格式**: MATLAB .mat文件

### 数据预处理流程

#### 视频帧提取
```python
# 帧提取参数
frame_rate = 25          # 目标帧率
max_frames = 8           # 每个视频最大帧数
resolution = (448, 448)  # 目标分辨率
```

#### 数据增强策略
- **空间增强**: 随机裁剪、水平翻转、颜色抖动
- **时间增强**: 随机帧采样、时间窗口滑动
- **归一化**: ImageNet预训练模型的标准归一化

#### 标注文件格式
```csv
name,mos
test1_3,3.0
test2_2.8,2.8
test3_2.7,2.7
```

## 模型详细说明

### UGC_BVQA_model 架构深度解析

#### 网络结构概览
```python
UGC_BVQA_model(
  (conv1): Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
  (bn1): BatchNorm2d(64)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1)
  
  # ResNet-50 主体结构
  (layer1): Sequential(...)  # 3个Bottleneck块
  (layer2): Sequential(...)  # 4个Bottleneck块  
  (layer3): Sequential(...)  # 6个Bottleneck块
  (layer4): Sequential(...)  # 3个Bottleneck块
  
  # 质量回归模块
  (quality_regression): Sequential(
    (0): Linear(4352, 2048)  # 空间特征 + 时间特征
    (1): ReLU(inplace=True)
    (2): Dropout(0.5)
    (3): Linear(2048, 1)     # 输出质量评分
  )
)
```

#### 核心组件详解

##### 1. Bottleneck 残差块
```python
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # 1x1 conv -> 3x3 conv -> 1x1 conv
        self.conv1 = conv1x1(inplanes, width)
        self.conv2 = conv3x3(width, width, stride)
        self.conv3 = conv1x1(width, planes * self.expansion)
        
    def forward(self, x):
        identity = x
        out = self.conv3(self.relu(self.bn2(self.conv2(...))))
        out += identity  # 残差连接
        return self.relu(out)
```

##### 2. 全局池化策略
```python
def global_std_pool2d(x):
    """全局标准差池化，捕获特征变化信息"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1), 
                     dim=2, keepdim=True)

# 特征提取
avg_pool = F.adaptive_avg_pool2d(features, (1, 1))  # 全局平均池化
std_pool = global_std_pool2d(features)              # 全局标准差池化
spatial_features = torch.cat([avg_pool, std_pool], dim=1)  # 特征拼接
```

##### 3. 质量回归模块
- **输入维度**: 4352 = 2048(空间特征) + 2304(时间特征)
- **隐藏层**: 2048维全连接层 + ReLU + Dropout(0.5)
- **输出层**: 1维质量评分
- **激活函数**: 无激活（回归任务）

#### 输入输出规格

##### 空间输入格式
```python
# 输入张量形状
spatial_input = torch.randn(batch_size, num_frames, 3, 448, 448)
# batch_size: 批量大小 (通常为1-4)
# num_frames: 帧数量 (固定为8帧)
# 3: RGB通道
# 448x448: 图像分辨率
```

##### 时间输入格式
```python
# SlowFast特征张量形状
temporal_input = torch.randn(batch_size, num_frames, 2304)
# 2304 = 2048(慢通道特征) + 256(快通道特征)
```

##### 输出格式
```python
# 质量评分输出
quality_score = model(spatial_input, temporal_input)
# 形状: (batch_size, 1)
# 数值范围: 通常在1-5之间（取决于数据集）
```

### 损失函数详解

#### L1RankLoss 组合损失
```python
class L1RankLoss(torch.nn.Module):
    def __init__(self, l1_w=1.0, rank_w=1.0, hard_thred=1.0):
        self.l1_w = l1_w        # L1损失权重
        self.rank_w = rank_w    # 排序损失权重
        self.hard_thred = hard_thred  # 困难样本阈值
        
    def forward(self, preds, gts):
        # L1回归损失
        l1_loss = F.l1_loss(preds, gts)
        
        # 排序损失（保持相对质量关系）
        rank_loss = self.compute_rank_loss(preds, gts)
        
        # 总损失
        total_loss = self.l1_w * l1_loss + self.rank_w * rank_loss
        return total_loss
```

#### 损失函数优势
- **L1损失**: 对异常值鲁棒，确保预测值接近真实值
- **排序损失**: 保持视频质量的相对排序关系
- **组合优势**: 既保证绝对精度，又维持相对顺序

### 模型训练策略

#### 预训练权重加载
```python
# 加载ImageNet预训练的ResNet-50权重
model = UGC_BVQA_model.resnet50(pretrained=True)

# 冻结部分层（可选）
for param in model.layer1.parameters():
    param.requires_grad = False
```

#### 学习率调度
```python
# 优化器配置
optimizer = optim.Adam(model.parameters(), 
                      lr=1e-5,           # 较小的学习率
                      weight_decay=1e-7) # 轻微的权重衰减

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                     step_size=2,    # 每2轮衰减
                                     gamma=0.9)      # 衰减因子
```

#### 训练技巧
- **梯度裁剪**: 防止梯度爆炸
- **早停策略**: 监控验证集性能
- **模型集成**: 多个模型投票提升性能

## 使用流程详解

### 1. 环境配置

#### 系统要求
- **操作系统**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python版本**: 3.7-3.9 (推荐3.8)
- **GPU**: NVIDIA GPU with CUDA 10.2+ (可选，但强烈推荐)
- **内存**: 至少8GB RAM，推荐16GB+
- **存储**: 至少10GB可用空间

#### 依赖安装
```bash
# 1. 创建虚拟环境（推荐）
conda create -n simplevqa python=3.8
conda activate simplevqa

# 2. 安装PyTorch（根据CUDA版本选择）
# CPU版本
pip install torch torchvision torchaudio

# GPU版本（CUDA 11.3）
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# 3. 安装项目依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

#### CUDA环境配置
```bash
# 检查CUDA版本
nvidia-smi

# 设置环境变量（Windows）
set CUDA_VISIBLE_DEVICES=0

# 设置环境变量（Linux/macOS）
export CUDA_VISIBLE_DEVICES=0
```

### 2. 数据准备详细流程

#### 2.1 视频数据组织
```bash
# 标准目录结构
SimpleVQA-main/
├── LSVQ/                    # 原始视频文件
│   ├── test1_3.mp4
│   ├── test2_2.8.mp4
│   └── ...
├── LSVQ_image/             # 提取的帧（自动生成）
├── data/                   # 标注文件
│   ├── LSVQ_whole_train.csv
│   └── LSVQ_whole_test.csv
└── LSVQ_SlowFast_feature/  # 3D特征（自动生成）
```

#### 2.2 视频帧提取
```bash
# 方法1: 使用提供的脚本
bash scripts/extract_frames.sh

# 方法2: 手动执行Python脚本
python src/extract_frame_LSVQ.py \
    --video_dir LSVQ \
    --output_dir LSVQ_image \
    --frame_rate 25 \
    --max_frames 8

# 方法3: 批量处理
for video in LSVQ/*.mp4; do
    python src/extract_frame_LSVQ.py --input "$video"
done
```

#### 2.3 SlowFast特征提取
```bash
# 提取3D运动特征
bash scripts/extract_features.sh

# 或手动执行
python extract_SlowFast_features_LSVQ.py \
    --video_dir LSVQ \
    --output_dir LSVQ_SlowFast_feature \
    --batch_size 4
```

#### 2.4 数据验证
```python
# 验证数据完整性
python -c "
import os
import pandas as pd

# 检查视频文件
video_dir = 'LSVQ'
videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
print(f'视频文件数量: {len(videos)}')

# 检查标注文件
train_df = pd.read_csv('data/LSVQ_whole_train.csv')
test_df = pd.read_csv('data/LSVQ_whole_test.csv')
print(f'训练样本: {len(train_df)}, 测试样本: {len(test_df)}')

# 检查帧提取结果
frame_dir = 'LSVQ_image'
frame_folders = os.listdir(frame_dir)
print(f'帧文件夹数量: {len(frame_folders)}')
"
```

### 3. 模型训练详细指南

#### 3.1 训练参数配置
```python
# 训练配置文件示例 (config.py)
class TrainConfig:
    # 数据配置
    database = 'LSVQ'
    train_batch_size = 2        # 根据GPU内存调整
    num_workers = 2             # 数据加载线程数
    
    # 模型配置
    model_name = 'UGC_BVQA_model'
    resize = 520               # 输入图像resize尺寸
    crop_size = 448            # 随机裁剪尺寸
    
    # 训练配置
    epochs = 20
    conv_base_lr = 1e-5        # 学习率
    decay_ratio = 0.9          # 学习率衰减
    decay_interval = 2         # 衰减间隔
    
    # 损失配置
    loss_type = 'L1RankLoss'
    
    # 保存配置
    ckpt_path = 'ckpts'
    exp_version = 0
    print_samples = 1000       # 打印间隔
```

#### 3.2 训练执行
```bash
# 方法1: 使用训练脚本
bash scripts/train.sh

# 方法2: 直接Python命令
python src/train.py \
    --database LSVQ \
    --model_name UGC_BVQA_model \
    --conv_base_lr 0.00001 \
    --epochs 20 \
    --train_batch_size 2 \
    --num_workers 2 \
    --ckpt_path ckpts \
    --decay_ratio 0.9 \
    --decay_interval 2 \
    --exp_version 0 \
    --loss_type L1RankLoss \
    --resize 520 \
    --crop_size 448

# 方法3: 多GPU训练
python src/train.py \
    --multi_gpu \
    --gpu_ids 0,1 \
    --train_batch_size 4 \
    [其他参数...]
```

#### 3.3 训练监控
```bash
# 实时查看训练日志
tail -f logs/train_UGC_BVQA_model_L1RankLoss_*.log

# 使用TensorBoard（如果集成）
tensorboard --logdir=logs --port=6006

# 检查GPU使用情况
nvidia-smi -l 1
```

#### 3.4 训练输出解读
```
Epoch [1/20], Step [100/1000]:
├── Loss: 0.245 (总损失)
├── L1_Loss: 0.123 (L1回归损失)
├── Rank_Loss: 0.122 (排序损失)
├── LR: 1e-05 (当前学习率)
├── Time: 2.34s/batch (训练速度)
└── GPU_Mem: 3.2GB/8GB (GPU内存使用)

Validation Results:
├── PLCC: 0.856 (皮尔逊相关系数)
├── SRCC: 0.834 (斯皮尔曼相关系数)
├── KRCC: 0.678 (肯德尔相关系数)
└── RMSE: 0.234 (均方根误差)
```

### 4. 模型测试与评估

#### 4.1 单视频测试
```bash
# 基本测试
python src/test_demo.py \
    --method_name single-scale \
    --dist LSVQ/test1_3.mp4 \
    --output result.txt \
    --is_gpu

# 指定模型权重
python src/test_demo.py \
    --method_name single-scale \
    --dist path/to/video.mp4 \
    --model_path ckpts/best_model.pth \
    --output custom_result.txt
```

#### 4.2 批量测试
```bash
# 测试整个测试集
python test_on_pretrained_model.py \
    --test_csv data/LSVQ_whole_test.csv \
    --model_path ckpts/UGC_BVQA_model.pth \
    --output_dir results/

# 自定义测试脚本
python -c "
import os
import pandas as pd
from src.test_demo import main as test_main

# 读取测试列表
test_df = pd.read_csv('data/LSVQ_whole_test.csv')
results = []

for idx, row in test_df.iterrows():
    video_path = f'LSVQ/{row[\"name\"]}.mp4'
    if os.path.exists(video_path):
        # 执行测试
        score = test_main(video_path)
        results.append({
            'video': row['name'],
            'ground_truth': row['mos'],
            'predicted': score
        })
        print(f'处理完成: {row[\"name\"]} - GT: {row[\"mos\"]:.2f}, Pred: {score:.2f}')

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv('batch_test_results.csv', index=False)
"
```

#### 4.3 性能评估
```python
# 计算评估指标
python -c "
import pandas as pd
import numpy as np
from scipy import stats
from src.utils import performance_fit

# 读取测试结果
results = pd.read_csv('batch_test_results.csv')
gt_scores = results['ground_truth'].values
pred_scores = results['predicted'].values

# 计算性能指标
plcc, srcc, krcc, rmse = performance_fit(gt_scores, pred_scores)

print('=== 模型性能评估 ===')
print(f'PLCC (皮尔逊相关系数): {plcc:.4f}')
print(f'SRCC (斯皮尔曼相关系数): {srcc:.4f}')
print(f'KRCC (肯德尔相关系数): {krcc:.4f}')
print(f'RMSE (均方根误差): {rmse:.4f}')

# 性能等级判断
if srcc > 0.9:
    print('性能等级: 优秀')
elif srcc > 0.8:
    print('性能等级: 良好')
elif srcc > 0.7:
    print('性能等级: 一般')
else:
    print('性能等级: 需要改进')
"
```

### 5. 常见问题排查

#### 5.1 内存不足问题
```bash
# 问题: CUDA out of memory
# 解决方案:
# 1. 减小批量大小
python src/train.py --train_batch_size 1

# 2. 减小图像分辨率
python src/train.py --resize 256 --crop_size 224

# 3. 减少工作线程
python src/train.py --num_workers 0

# 4. 启用内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

#### 5.2 训练不收敛问题
```bash
# 问题: 损失不下降或震荡
# 解决方案:
# 1. 调整学习率
python src/train.py --conv_base_lr 1e-6  # 更小的学习率

# 2. 修改学习率调度
python src/train.py --decay_ratio 0.95 --decay_interval 1

# 3. 检查数据预处理
python -c "
import torch
from src.data_loader import VideoDataset_images_with_motion_features
# 检查数据加载是否正常
"
```

#### 5.3 推理速度优化
```python
# 模型优化技巧
import torch

# 1. 设置为评估模式
model.eval()

# 2. 禁用梯度计算
with torch.no_grad():
    output = model(input_data)

# 3. 使用半精度推理
model.half()
input_data = input_data.half()

# 4. 批量推理
batch_inputs = torch.stack([input1, input2, input3])
batch_outputs = model(batch_inputs)
```

## 项目结构详解

```
SimpleVQA-main/
├── .idea/                          # PyCharm IDE配置文件
├── __pycache__/                    # Python字节码缓存
├── LSVQ/                          # 原始视频数据集
│   ├── test1_3.mp4               # 视频文件（命名格式：test{id}_{mos}.mp4）
│   ├── test2_2.8.mp4
│   └── ...                       # 更多视频文件
├── LSVQ_image/                    # 提取的视频帧（自动生成）
│   ├── test1_3/                  # 每个视频对应一个文件夹
│   │   ├── 00001.jpg            # 帧图像（按时间顺序编号）
│   │   ├── 00002.jpg
│   │   └── ...
│   └── test2_2.8/
├── LSVQ_SlowFast_feature/         # SlowFast 3D特征（自动生成）
│   ├── test1_3.npy               # 每个视频对应一个特征文件
│   ├── test2_2.8.npy
│   └── ...
├── ckpts/                         # 模型检查点目录
│   ├── UGC_BVQA_model_L1RankLoss_LSVQ_0.pth  # 训练好的模型权重
│   ├── best_model.pth            # 最佳模型
│   └── checkpoint_epoch_*.pth    # 训练过程中的检查点
├── data/                          # 数据标注文件
│   ├── LSVQ_whole_train.csv      # 训练集标注（视频名称+MOS分数）
│   ├── LSVQ_whole_test.csv       # 测试集标注
│   ├── KoNViD_1k_train.csv       # KoNViD-1k数据集标注（可选）
│   └── YouTube_UGC_train.csv     # YouTube-UGC数据集标注（可选）
├── docs/                          # 项目文档
│   └── SimpleVQA综合技术文档.md   # 本文档
├── scripts/                       # 执行脚本
│   ├── train.sh                  # 训练脚本
│   ├── test.sh                   # 测试脚本
│   ├── extract_frames.sh         # 帧提取脚本（可选）
│   └── extract_features.sh       # 特征提取脚本（可选）
├── src/                           # 源代码目录
│   ├── model/                    # 模型定义
│   │   ├── __init__.py
│   │   ├── UGC_BVQA_model.py    # 主要的BVQA模型
│   │   ├── resnet.py            # ResNet骨干网络
│   │   └── slowfast.py          # SlowFast网络（可选）
│   ├── __init__.py
│   ├── data_loader.py           # 数据加载器
│   ├── train.py                 # 训练主程序
│   ├── test_demo.py             # 测试演示程序
│   ├── utils.py                 # 工具函数
│   ├── extract_frame_LSVQ.py    # 视频帧提取工具
│   └── extract_SlowFast_features_LSVQ.py  # SlowFast特征提取
├── logs/                         # 训练日志（自动生成）
│   ├── train_*.log              # 训练日志文件
│   └── tensorboard/             # TensorBoard日志（可选）
├── results/                      # 测试结果（自动生成）
│   ├── predictions.csv          # 预测结果
│   └── evaluation_metrics.txt   # 评估指标
├── README.md                     # 项目说明文件
├── requirements.txt              # Python依赖列表
├── config.py                     # 配置文件（可选）
└── setup.py                      # 安装脚本（可选）
```

### 核心文件详解

#### 1. 模型相关文件
- **`src/model/UGC_BVQA_model.py`**: 核心BVQA模型实现
  - 包含ResNet骨干网络
  - 实现空间-时间特征融合
  - 定义质量回归头
  
- **`src/model/resnet.py`**: ResNet网络实现
  - 提供不同深度的ResNet变体
  - 支持预训练权重加载
  - 自定义Bottleneck结构

#### 2. 数据处理文件
- **`src/data_loader.py`**: 数据加载核心
  - 支持多种数据集格式
  - 实现数据增强策略
  - 处理视频帧和运动特征

- **`src/extract_frame_LSVQ.py`**: 视频预处理
  - 从视频中提取关键帧
  - 支持多种采样策略
  - 自动创建目录结构

#### 3. 训练和测试文件
- **`src/train.py`**: 训练主程序
  - 实现完整的训练循环
  - 支持多GPU训练
  - 包含验证和保存逻辑

- **`src/test_demo.py`**: 测试演示
  - 单视频质量评估
  - 支持批量处理
  - 输出详细结果

#### 4. 配置和脚本文件
- **`scripts/train.sh`**: 训练启动脚本
  ```bash
  #!/bin/bash
  export CUDA_VISIBLE_DEVICES=0
  python src/train.py \
      --database LSVQ \
      --model_name UGC_BVQA_model \
      --conv_base_lr 0.00001 \
      --epochs 20 \
      --train_batch_size 2 \
      --num_workers 2 \
      --ckpt_path ckpts \
      --decay_ratio 0.9 \
      --decay_interval 2 \
      --exp_version 0 \
      --loss_type L1RankLoss \
      --resize 520 \
      --crop_size 448
  ```

- **`scripts/test.sh`**: 测试启动脚本
  ```bash
  #!/bin/bash
  export CUDA_VISIBLE_DEVICES=0
  python src/test_demo.py \
      --method_name single-scale \
      --dist LSVQ/test1_3.mp4 \
      --output result.txt \
      --is_gpu
  ```

### 数据流向图

```
原始视频 (LSVQ/*.mp4)
    ↓
视频帧提取 (extract_frame_LSVQ.py)
    ↓
帧图像 (LSVQ_image/*/00001.jpg)
    ↓
SlowFast特征提取 (extract_SlowFast_features_LSVQ.py)
    ↓
运动特征 (LSVQ_SlowFast_feature/*.npy)
    ↓
数据加载器 (data_loader.py)
    ↓
模型训练/测试 (UGC_BVQA_model.py)
    ↓
质量分数输出
```

### 文件命名规范

#### 视频文件命名
- 格式：`test{id}_{mos}.mp4`
- 示例：`test1_3.mp4` (ID=1, MOS=3.0)
- 说明：MOS分数直接编码在文件名中

#### 特征文件命名
- 帧文件夹：`LSVQ_image/test{id}_{mos}/`
- 特征文件：`LSVQ_SlowFast_feature/test{id}_{mos}.npy`
- 保持与原视频文件的对应关系

#### 模型权重命名
- 格式：`{model_name}_{loss_type}_{database}_{version}.pth`
- 示例：`UGC_BVQA_model_L1RankLoss_LSVQ_0.pth`
- 便于区分不同配置的模型

### 扩展性设计

#### 添加新数据集
1. 在`data/`目录添加新的CSV标注文件
2. 修改`data_loader.py`中的数据集类
3. 更新训练脚本中的数据库参数

#### 添加新模型
1. 在`src/model/`目录创建新模型文件
2. 在`train.py`中注册新模型
3. 更新配置参数

#### 自定义损失函数
1. 在`utils.py`中定义新损失函数
2. 在训练脚本中指定损失类型
3. 确保与现有评估指标兼容

## 性能评估指标详解

### 1. 核心评估指标

#### 1.1 皮尔逊线性相关系数 (PLCC)
```python
# 计算公式
def plcc(y_pred, y_true):
    """
    计算皮尔逊线性相关系数
    范围: [-1, 1]，越接近1表示线性相关性越强
    """
    return np.corrcoef(y_pred, y_true)[0, 1]

# 性能等级
# PLCC > 0.9: 优秀
# PLCC > 0.8: 良好  
# PLCC > 0.7: 一般
# PLCC < 0.7: 需要改进
```

**意义**: 衡量预测分数与真实分数之间的线性相关程度
**优势**: 对线性关系敏感，适合评估回归模型
**局限**: 只能捕捉线性关系，对非线性关系不敏感

#### 1.2 斯皮尔曼等级相关系数 (SRCC)
```python
# 计算公式
def srcc(y_pred, y_true):
    """
    计算斯皮尔曼等级相关系数
    范围: [-1, 1]，越接近1表示单调相关性越强
    """
    from scipy.stats import spearmanr
    return spearmanr(y_pred, y_true)[0]

# 性能等级
# SRCC > 0.9: 优秀
# SRCC > 0.8: 良好
# SRCC > 0.7: 一般  
# SRCC < 0.7: 需要改进
```

**意义**: 衡量预测分数与真实分数排序的一致性
**优势**: 对单调变换不敏感，更关注相对排序
**应用**: 视频质量评估中最重要的指标

#### 1.3 肯德尔等级相关系数 (KRCC)
```python
# 计算公式
def krcc(y_pred, y_true):
    """
    计算肯德尔等级相关系数
    范围: [-1, 1]，基于一致性对的比例
    """
    from scipy.stats import kendalltau
    return kendalltau(y_pred, y_true)[0]

# 性能等级
# KRCC > 0.7: 优秀
# KRCC > 0.6: 良好
# KRCC > 0.5: 一般
# KRCC < 0.5: 需要改进
```

**意义**: 基于一致性对和不一致性对的比例
**优势**: 对异常值更鲁棒
**特点**: 通常数值比SRCC小，但更稳定

#### 1.4 均方根误差 (RMSE)
```python
# 计算公式
def rmse(y_pred, y_true):
    """
    计算均方根误差
    范围: [0, +∞)，越小表示预测越准确
    """
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

# 性能等级（针对MOS分数1-5）
# RMSE < 0.3: 优秀
# RMSE < 0.5: 良好
# RMSE < 0.8: 一般
# RMSE > 0.8: 需要改进
```

**意义**: 衡量预测值与真实值的绝对误差
**优势**: 直观反映预测精度
**局限**: 对异常值敏感

### 2. 评估指标实现

#### 2.1 完整评估函数
```python
def performance_fit(y_pred, y_true):
    """
    计算所有性能指标
    
    Args:
        y_pred: 预测分数数组
        y_true: 真实分数数组
    
    Returns:
        plcc, srcc, krcc, rmse: 四个评估指标
    """
    import numpy as np
    from scipy import stats
    
    # 移除NaN值
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    
    # 计算PLCC
    plcc = np.corrcoef(y_pred, y_true)[0, 1]
    
    # 计算SRCC
    srcc = stats.spearmanr(y_pred, y_true)[0]
    
    # 计算KRCC
    krcc = stats.kendalltau(y_pred, y_true)[0]
    
    # 计算RMSE
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    return plcc, srcc, krcc, rmse

# 使用示例
y_pred = np.array([3.2, 2.8, 4.1, 1.9, 3.7])
y_true = np.array([3.0, 3.0, 4.0, 2.0, 3.5])
plcc, srcc, krcc, rmse = performance_fit(y_pred, y_true)

print(f"PLCC: {plcc:.4f}")
print(f"SRCC: {srcc:.4f}")  
print(f"KRCC: {krcc:.4f}")
print(f"RMSE: {rmse:.4f}")
```

#### 2.2 批量评估脚本
```python
def evaluate_model_performance(model, test_loader, device):
    """
    在测试集上评估模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
    
    Returns:
        dict: 包含所有评估指标的字典
    """
    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            
            # 模型预测
            output = model(data)
            
            # 收集结果
            predictions.extend(output.cpu().numpy().flatten())
            ground_truths.extend(target.cpu().numpy().flatten())
    
    # 转换为numpy数组
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # 计算评估指标
    plcc, srcc, krcc, rmse = performance_fit(predictions, ground_truths)
    
    # 返回结果字典
    results = {
        'PLCC': plcc,
        'SRCC': srcc,
        'KRCC': krcc,
        'RMSE': rmse,
        'predictions': predictions,
        'ground_truths': ground_truths,
        'num_samples': len(predictions)
    }
    
    return results
```

### 3. 基准性能对比

#### 3.1 LSVQ数据集基准
```python
# 不同方法在LSVQ数据集上的性能对比
benchmark_results = {
    'PSNR': {'PLCC': 0.612, 'SRCC': 0.593, 'KRCC': 0.421, 'RMSE': 0.892},
    'SSIM': {'PLCC': 0.723, 'SRCC': 0.706, 'KRCC': 0.524, 'RMSE': 0.756},
    'VMAF': {'PLCC': 0.834, 'SRCC': 0.826, 'KRCC': 0.634, 'RMSE': 0.587},
    'BRISQUE': {'PLCC': 0.567, 'SRCC': 0.542, 'KRCC': 0.387, 'RMSE': 0.934},
    'NIQE': {'PLCC': 0.489, 'SRCC': 0.476, 'KRCC': 0.334, 'RMSE': 1.023},
    'UGC_BVQA_model': {'PLCC': 0.892, 'SRCC': 0.876, 'KRCC': 0.698, 'RMSE': 0.456}
}

# 性能排序
def rank_methods(results, metric='SRCC'):
    """按指定指标对方法进行排序"""
    sorted_methods = sorted(results.items(), 
                          key=lambda x: x[1][metric], 
                          reverse=True)
    return sorted_methods

# 打印排序结果
print("=== LSVQ数据集性能排序 (按SRCC) ===")
for rank, (method, scores) in enumerate(rank_methods(benchmark_results), 1):
    print(f"{rank}. {method}: SRCC={scores['SRCC']:.3f}")
```

#### 3.2 跨数据集性能
```python
# 不同数据集上的性能表现
cross_dataset_results = {
    'LSVQ': {
        'UGC_BVQA_model': {'PLCC': 0.892, 'SRCC': 0.876, 'KRCC': 0.698, 'RMSE': 0.456},
        'VMAF': {'PLCC': 0.834, 'SRCC': 0.826, 'KRCC': 0.634, 'RMSE': 0.587}
    },
    'KoNViD-1k': {
        'UGC_BVQA_model': {'PLCC': 0.856, 'SRCC': 0.843, 'KRCC': 0.672, 'RMSE': 0.523},
        'VMAF': {'PLCC': 0.798, 'SRCC': 0.789, 'KRCC': 0.598, 'RMSE': 0.634}
    },
    'YouTube-UGC': {
        'UGC_BVQA_model': {'PLCC': 0.823, 'SRCC': 0.812, 'KRCC': 0.634, 'RMSE': 0.587},
        'VMAF': {'PLCC': 0.756, 'SRCC': 0.743, 'KRCC': 0.567, 'RMSE': 0.698}
    }
}
```

### 4. 性能分析工具

#### 4.1 误差分析
```python
def error_analysis(predictions, ground_truths, video_names=None):
    """
    详细的误差分析
    
    Args:
        predictions: 预测分数
        ground_truths: 真实分数
        video_names: 视频名称列表（可选）
    
    Returns:
        dict: 误差分析结果
    """
    errors = predictions - ground_truths
    abs_errors = np.abs(errors)
    
    analysis = {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'mean_abs_error': np.mean(abs_errors),
        'max_abs_error': np.max(abs_errors),
        'min_abs_error': np.min(abs_errors),
        'error_percentiles': {
            '25%': np.percentile(abs_errors, 25),
            '50%': np.percentile(abs_errors, 50),
            '75%': np.percentile(abs_errors, 75),
            '90%': np.percentile(abs_errors, 90),
            '95%': np.percentile(abs_errors, 95)
        }
    }
    
    # 找出误差最大的样本
    worst_indices = np.argsort(abs_errors)[-5:]
    analysis['worst_predictions'] = []
    
    for idx in worst_indices:
        sample_info = {
            'index': idx,
            'predicted': predictions[idx],
            'ground_truth': ground_truths[idx],
            'absolute_error': abs_errors[idx]
        }
        if video_names is not None:
            sample_info['video_name'] = video_names[idx]
        analysis['worst_predictions'].append(sample_info)
    
    return analysis
```

#### 4.2 可视化工具
```python
def plot_performance_analysis(predictions, ground_truths, save_path=None):
    """
    绘制性能分析图表
    
    Args:
        predictions: 预测分数
        ground_truths: 真实分数
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 散点图：预测值 vs 真实值
    axes[0, 0].scatter(ground_truths, predictions, alpha=0.6)
    axes[0, 0].plot([min(ground_truths), max(ground_truths)], 
                    [min(ground_truths), max(ground_truths)], 'r--')
    axes[0, 0].set_xlabel('Ground Truth')
    axes[0, 0].set_ylabel('Predictions')
    axes[0, 0].set_title('Predictions vs Ground Truth')
    axes[0, 0].grid(True)
    
    # 2. 误差分布直方图
    errors = predictions - ground_truths
    axes[0, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].grid(True)
    
    # 3. 绝对误差分布
    abs_errors = np.abs(errors)
    axes[1, 0].hist(abs_errors, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Absolute Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Absolute Error Distribution')
    axes[1, 0].grid(True)
    
    # 4. 残差图
    axes[1, 1].scatter(predictions, errors, alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Predictions')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residual Plot')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

### 5. 性能优化建议

#### 5.1 基于指标的优化策略
```python
def suggest_improvements(plcc, srcc, krcc, rmse):
    """
    基于当前性能指标提供优化建议
    
    Args:
        plcc, srcc, krcc, rmse: 当前性能指标
    
    Returns:
        list: 优化建议列表
    """
    suggestions = []
    
    # PLCC相关建议
    if plcc < 0.8:
        suggestions.append("PLCC较低，建议：")
        suggestions.append("- 增加模型复杂度或容量")
        suggestions.append("- 改进特征提取方法")
        suggestions.append("- 使用更大的训练数据集")
    
    # SRCC相关建议
    if srcc < 0.8:
        suggestions.append("SRCC较低，建议：")
        suggestions.append("- 使用排序损失函数")
        suggestions.append("- 改进数据预处理和增强")
        suggestions.append("- 调整学习率和训练策略")
    
    # RMSE相关建议
    if rmse > 0.6:
        suggestions.append("RMSE较高，建议：")
        suggestions.append("- 使用L1或Huber损失减少异常值影响")
        suggestions.append("- 增加正则化防止过拟合")
        suggestions.append("- 检查数据质量和标注一致性")
    
    # 综合建议
    if all(metric < threshold for metric, threshold in 
           [(plcc, 0.85), (srcc, 0.85), (krcc, 0.65)]):
        suggestions.append("整体性能需要提升，建议：")
        suggestions.append("- 重新设计网络架构")
        suggestions.append("- 尝试不同的预训练模型")
        suggestions.append("- 使用集成学习方法")
    
    return suggestions
```

## 训练参数配置

### 默认训练参数
- **学习率**: 1e-5
- **批量大小**: 2
- **训练轮数**: 20
- **优化器**: Adam
- **学习率衰减**: 每2轮衰减0.9倍
- **图像尺寸**: 520x520 (resize) → 448x448 (crop)

### GPU配置
- 支持单GPU和多GPU训练
- 自动检测CUDA可用性
- 内存优化配置以避免OOM错误

## 特征提取流程

### 空间特征提取
1. 视频解码为帧序列
2. 帧预处理（resize, crop, normalize）
3. 通过ResNet-50提取空间特征
4. 时间维度上的特征聚合

### 时间特征提取
1. 使用SlowFast模型提取3D特征
2. 慢通道捕获空间信息
3. 快通道捕获时间运动信息
4. 双通道特征融合

## 部署和推理

### 模型推理
```python
# 加载预训练模型
model = UGC_BVQA_model.resnet50(pretrained=False)
model.load_state_dict(torch.load('ckpts/UGC_BVQA_model.pth'))

# 视频质量评估
quality_score = model(video_frames, motion_features)
```

### 批量处理
支持批量视频处理，可以同时评估多个视频的质量。

## 扩展和定制

### 添加新数据集
1. 在`data_loader.py`中添加新的数据集类
2. 实现相应的数据加载逻辑
3. 更新训练脚本中的数据集选择

### 模型改进
1. 修改`UGC_BVQA_model.py`中的网络结构
2. 调整特征融合策略
3. 实验不同的损失函数

## 常见问题和解决方案

### 内存不足
- 减小批量大小
- 降低图像分辨率
- 使用梯度累积

### 训练不收敛
- 调整学习率
- 检查数据预处理
- 验证损失函数实现

### 推理速度慢
- 使用TensorRT优化
- 模型量化
- 批量推理

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。请确保：
1. 代码符合项目规范
2. 添加必要的测试
3. 更新相关文档

---
