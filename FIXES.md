# 模型训练问题修复方案

## 问题分析

### 原始问题
1. **LSTM模型准确率**: 25%
2. **MLP模型准确率**: 3%
3. **Transformer模型准确率**: 0.8%
4. **Loss**: NaN (数值不稳定)

### 根本原因
1. **特征尺度不匹配**:
   - `buffer_bytes`: VO(200-4000), VI(400-20000), BE(400-1500000) - 范围差异巨大
   - `wait_ms`: VO(0-40), VI(0-80), BE(0-150) - 与buffer_bytes的数量级完全不同
   - 未归一化的原始数据导致梯度不稳定

2. **模型架构问题**:
   - 缺少LayerNorm导致内部激活值范围不稳定
   - Transformer使用post-LN架构，训练不稳定
   - 初始化权重可能过大

3. **训练超参数**:
   - 学习率1e-3对于归一化前的数据过大
   - Batch size 1024可能导致梯度估计不准确

## 解决方案

### 1. 数据归一化 (dataset.py)
```python
# 按业务类型分别归一化
buffer_bytes[VO] /= 4000
buffer_bytes[VI] /= 20000
buffer_bytes[BE] /= 1500000

wait_ms[VO] /= 40
wait_ms[VI] /= 80
wait_ms[BE] /= 150
```

**效果**: 所有特征值归一化到[0, 1]范围，消除数值不稳定

### 2. 模型架构改进

#### BiLSTM
- ✅ 添加输入LayerNorm
- ✅ 输出层添加LayerNorm
- ✅ 增加Dropout
- ✅ 使用xavier初始化(gain=0.1)

#### MLP
- ✅ 添加输入LayerNorm
- ✅ 每层后添加LayerNorm
- ✅ 改进残差连接

#### Transformer
- ✅ 添加输入LayerNorm
- ✅ 使用Pre-LN架构(norm_first=True)
- ✅ 增加中间层和Dropout
- ✅ 使用xavier初始化(gain=0.1)

### 3. 训练策略优化 (train.py)

#### 超参数调整
```python
learning_rate: 1e-3 → 1e-4  (降低10倍)
batch_size: 1024 → 256      (减小4倍)
```

#### 添加学习率调度器
```python
CosineAnnealingLR(T_max=epochs, eta_min=lr*0.01)
```

#### 添加NaN检测和保护
- 检查输入特征
- 检查模型输出
- 检查loss值
- 跳过异常batch

#### 改进权重初始化
```python
Linear: xavier_uniform (gain=0.1)
LSTM: xavier_uniform for input weights, orthogonal for hidden weights
Bias: constant(0)
```

## 使用方法

### 1. 诊断现有数据
```bash
python diagnose_data.py --data data/train.npz
```

### 2. 快速测试修复效果
```bash
python quick_test.py
```

### 3. 训练模型

#### BiLSTM (推荐先尝试)
```bash
python train.py --model bilstm --lr 1e-4 --batch-size 256 --epochs 50
```

#### MLP
```bash
python train.py --model mlp --lr 1e-4 --batch-size 256 --epochs 50
```

#### Transformer
```bash
python train.py --model transformer --lr 5e-5 --batch-size 128 --epochs 50
```

## 预期改进

### 训练稳定性
- ❌ Loss: NaN
- ✅ Loss: 正常数值，稳定下降

### 准确率提升
- LSTM: 25% → 预期 60-80%
- MLP: 3% → 预期 40-60%
- Transformer: 0.8% → 预期 50-70%

## 进一步优化建议

如果准确率仍不理想，可以尝试:

1. **增加模型容量**
   ```bash
   # Transformer with larger model
   python train.py --model transformer --lr 3e-5 --batch-size 64
   ```

2. **数据增强**
   - 添加噪声扰动
   - 随机mask某些用户特征

3. **集成学习**
   - 训练多个模型取平均

4. **调整损失函数**
   - 使用Label Smoothing
   - 添加辅助损失(预测service类型)

5. **特征工程**
   - 添加衍生特征(buffer/wait比值)
   - 添加位置编码

## 故障排查

### 如果仍出现NaN
1. 检查数据: `python diagnose_data.py`
2. 进一步降低学习率: `--lr 1e-5`
3. 减小batch size: `--batch-size 64`
4. 增加梯度裁剪: 修改`max_norm=0.5`

### 如果准确率提升有限
1. 检查类别平衡性
2. 增加训练epoch
3. 尝试不同的模型架构
4. 检查标签质量

## 关键改动文件
- ✅ `dataset.py`: 添加特征归一化
- ✅ `model.py`: 改进三个模型架构
- ✅ `train.py`: 优化训练流程和超参数
- ✅ `diagnose_data.py`: 新增数据诊断工具
- ✅ `quick_test.py`: 新增快速测试工具
