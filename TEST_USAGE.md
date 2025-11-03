# 测试脚本使用说明

## 概述

新增了独立的测试脚本 `test.py`，用于加载训练好的模型并在指定测试集上进行评估。

## 主要改进

### 1. 归一化参数共享
- 测试集现在会使用训练集的归一化参数，确保数据处理一致性
- 归一化参数（`buf_max` 和 `wait_max`）会保存在模型检查点中

### 2. 场景名称标识
- 保存的模型现在包含训练场景和测试场景的名称
- 模型文件名和历史记录会包含场景信息

### 3. 独立测试脚本
- 可以在不重新训练的情况下测试模型
- 支持使用不同的测试集评估同一个模型

## 训练时使用不同的测试集

```powershell
python train.py --data data/train_scenario1.npz --test-data data/test_scenario2.npz --model bilstm --epochs 100
```

这将：
- 使用 `train_scenario1.npz` 作为训练数据
- 使用 `test_scenario2.npz` 作为测试数据
- 测试集会使用训练集的归一化参数
- 保存的模型包含场景信息

## 使用测试脚本

### 基本用法

```powershell
python test.py --checkpoint checkpoints/best.pt --test-data data/test_scenario3.npz
```

### 保存测试结果到JSON

```powershell
python test.py --checkpoint checkpoints/best.pt --test-data data/test_scenario3.npz --output results/test_results.json
```

### 使用混合精度加速推理

```powershell
python test.py --checkpoint checkpoints/best.pt --test-data data/test_scenario3.npz --amp
```

### 完整参数说明

- `--checkpoint`: 模型检查点路径（必需）
- `--test-data`: 测试数据集路径（必需）
- `--batch-size`: 批次大小（默认256）
- `--num-workers`: DataLoader工作线程数（默认4）
- `--amp`: 启用混合精度推理
- `--output`: 结果输出JSON文件路径（可选）

## 输出结果

测试脚本会显示：
- 模型信息（类型、标签模式、类别数等）
- 训练场景和测试场景
- 测试结果：
  - 损失（Loss）
  - Top-1 准确率
  - Top-5 准确率
  - Top-10 准确率

## 工作流示例

### 1. 转换多个CSV数据集为NPZ

```powershell
python convert_csv_to_npz.py --data-dir data/csv_files --label-mode combined
```

这会将文件夹中的所有 `input_data_*_set.csv` 和 `output_data_*_set.csv` 转换为对应的 `.npz` 文件。

### 2. 训练模型

```powershell
# 使用 mix 场景训练，valid 场景测试
python train.py --data data/vi_all_mix_set.npz --test-data data/vi_all_valid_set.npz --model bilstm --epochs 100 --save checkpoints/bilstm_vi_mix.pt
```

### 3. 在其他测试集上评估

```powershell
# 在 train 场景上测试
python test.py --checkpoint checkpoints/bilstm_vi_mix.pt --test-data data/vi_all_train_set.npz --output results/vi_mix_on_train.json

# 在 be 场景上测试
python test.py --checkpoint checkpoints/bilstm_vi_mix.pt --test-data data/be_all_mix_set.npz --output results/vi_mix_on_be.json
```

## 注意事项

1. **归一化参数一致性**：测试集会自动使用训练集的归一化参数，无需手动处理
2. **数据兼容性**：测试集必须与模型的标签模式、类别数和用户数匹配
3. **场景识别**：场景名称从文件名自动提取（去除 `.npz` 扩展名）
4. **结果保存**：使用 `--output` 参数可以将测试结果保存为JSON格式，便于后续分析
