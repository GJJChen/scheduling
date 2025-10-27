# Combined Label 模式支持文档

## 概述

本项目现在支持两种标签模式：
1. **User模式**：只预测用户ID (0-127)，共128个类别
2. **Combined模式**：预测用户×业务组合 (0-383)，共384个类别

## 数据转换

使用 `convert_csv_to_npz.py` 将CSV数据转换为npz格式：

```powershell
# Combined模式（默认）
python convert_csv_to_npz.py --input input.csv --output output.csv --out-npz data/train.npz --label-mode combined

# User模式
python convert_csv_to_npz.py --input input.csv --output output.csv --out-npz data/train.npz --label-mode user
```

### CSV文件格式

**输入CSV (input.csv)**:
- 第一行：序号（可选）
- 其他行：每行768个数值，表示展平的 [128, 3, 2] 张量

**输出CSV (output.csv)**:
- 第一行：序号（可选）
- 其他行：两列
  - 第一列：用户ID (0-127)
  - 第二列：业务ID (0-2)

### 标签编码

**Combined模式**:
```
y = user_id * 3 + service_id
范围: 0-383 (128用户 × 3业务)
```

**User模式**:
```
y = user_id
范围: 0-127
```

## 代码修改

### 1. dataset.py
- `SchedulingNPZDataset` 现在会自动检测标签模式
- 添加 `label_mode` 和 `num_classes` 属性
- 从npz文件的meta信息中读取标签模式

### 2. model.py
所有模型类（`BiLSTMClassifier`, `MLPClassifier`, `TransformerClassifier`）都添加了 `num_classes` 参数：

- **num_classes=128**: 原始模式，输出 [B, 128] 的logits
- **num_classes=384**: Combined模式，先展平序列特征，然后输出 [B, 384] 的logits

`build_model()` 函数签名变更：
```python
# 旧版本
build_model(name: str, input_dim=6)

# 新版本
build_model(name: str, input_dim=6, num_classes=128)
```

### 3. train.py
- 自动从数据集读取 `num_classes` 和 `label_mode`
- 使用正确的类别数构建模型
- 将标签信息保存到checkpoint中

### 4. test_improvements.py
- 更新测试函数以支持动态类别数

## 模型架构变化

### User模式 (num_classes=128)
```
输入 [B, 128, 6] 
  → 序列编码器 (LSTM/Transformer)
  → 每个用户位置输出一个分数
  → 输出 [B, 128]
```

### Combined模式 (num_classes=384)
```
输入 [B, 128, 6]
  → 序列编码器 (LSTM/Transformer)
  → 展平所有用户的特征 [B, 128*hidden]
  → 全连接层
  → 输出 [B, 384]
```

## 训练

训练时会自动检测数据集的标签模式：

```powershell
python train.py --data data/train.npz --model transformer --epochs 50
```

输出示例：
```
数据集标签模式: combined, 类别数: 384
模型: transformer, 输入维度: 6, 输出类别数: 384
```

## 性能考虑

### Combined模式的优势
1. 同时学习用户和业务的联合分布
2. 可以捕捉不同业务的调度偏好
3. 端到端优化

### Combined模式的挑战
1. 类别数增加到3倍（384 vs 128）
2. 模型参数量增加
3. 可能需要更多训练数据
4. 收敛速度可能变慢

### 建议
- 如果只关心用户选择，使用User模式（更简单、更快）
- 如果需要联合预测用户和业务，使用Combined模式
- Combined模式可能需要：
  - 更大的批量大小
  - 更长的训练时间
  - 更强的正则化

## 向后兼容性

旧的npz数据集（没有`label_mode`元数据）会自动识别为User模式，保持向后兼容。

## 检查点格式

新的检查点包含额外信息：
```python
{
    'model': 'transformer',
    'state_dict': {...},
    'cfg': {
        'N_USERS': 128,
        'num_classes': 384,  # 新增
        'label_mode': 'combined'  # 新增
    }
}
```

## 测试

运行测试以验证模型是否正常工作：
```powershell
python test_improvements.py
```

测试会自动适配数据集的类别数。
