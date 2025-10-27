# HierTransformer 分层模型训练修改说明

## 概述
为支持 `HierTransformer` 分层输出模型，对 `train.py` 进行了以下修改。

## HierTransformer 模型结构
- **输入**: `[B, 128, 6]` - Batch × 用户 × 特征
- **输出**: 
  - `service_logits`: `[B, 3]` - 业务类型预测（VO/VI/BE）
  - `user_logits`: `[B, 128]` - 用户预测

## 主要修改内容

### 1. `train_one_epoch` 函数修改
- **新增参数**: `is_hierarchical=False`
- **分层损失计算**:
  ```python
  # 从 combined label 解析 service 和 user
  service_tgt = label % 3        # 提取业务类型
  user_tgt = label // 3          # 提取用户ID
  
  # 计算两个子任务的损失
  loss_service = CrossEntropyLoss(service_logits, service_tgt)
  loss_user = CrossEntropyLoss(user_logits, user_tgt)
  loss = loss_service + loss_user  # 总损失
  ```
- **分层预测**:
  ```python
  pred_service = argmax(service_logits)  # 先预测业务类型
  pred_user = argmax(user_logits)        # 再预测用户
  pred = pred_user * 3 + pred_service    # 组合为最终预测
  ```

### 2. `evaluate` 函数修改
- **新增参数**: `is_hierarchical=False`
- **分层评估逻辑**: 与训练相同的损失和预测计算
- **Top-K 准确率**: 构建组合 logits 用于计算 Top-5/Top-10 准确率
  ```python
  combined_logits = torch.zeros(B, 384)  # 128 users × 3 services
  # 将 user_logits 和 service_logits 组合
  ```

### 3. `main` 函数修改
- **新增模型选项**: `--model hier` 支持选择 HierTransformer
- **模型初始化**:
  ```python
  if args.model == "hier":
      from model import HierTransformer
      model = HierTransformer(input_dim=6).to(device)
  ```
- **传递 is_hierarchical 标志**: 在训练和评估时传递该参数

## 使用方法

### 训练分层模型
```bash
python train.py --model hier --data data/train.npz --epochs 50 --batch-size 256 --lr 1e-2
```

### 训练普通模型（向后兼容）
```bash
python train.py --model transformer --data data/train.npz --epochs 50
python train.py --model bilstm --data data/train.npz --epochs 50
python train.py --model mlp --data data/train.npz --epochs 50
```

## 关键设计决策

### 1. 标签解析
- **Combined 模式标签格式**: `label = user_id * 3 + service_id`
- **解析方式**:
  - `service_id = label % 3` (0=VO, 1=VI, 2=BE)
  - `user_id = label // 3` (0-127)

### 2. 损失函数
- **简单相加**: `loss = loss_service + loss_user`
- **权重相同**: 两个子任务损失权重相等
- **可扩展**: 未来可调整为加权损失，如 `loss = α * loss_service + β * loss_user`

### 3. 预测策略
- **两阶段预测**:
  1. 独立预测 service (argmax on service_logits)
  2. 独立预测 user (argmax on user_logits)
  3. 组合为最终预测 (user * 3 + service)
- **注意**: 这与模型训练目标一致，但不同于"先选 service 再在该 service 下选 user"的策略

### 4. Top-K 准确率
- 构建 384 维组合 logits (128 users × 3 services)
- 简化处理：user_logits 复制到所有 service，加上 service_logits 的权重
- 用于计算 Top-5/Top-10 准确率

## 注意事项

1. **数据集要求**: 必须使用 `combined` 标签模式的数据集（384 类）
2. **向后兼容**: 所有修改都通过 `is_hierarchical` 参数控制，不影响现有模型
3. **GPU 内存**: HierTransformer 比单输出模型更轻量，显存占用较少
4. **性能**: 分层模型可能在某些场景下提供更好的可解释性和准确率

## 未来改进方向

1. **加权损失**: `loss = α * loss_service + β * loss_user`，可通过超参数调优
2. **条件预测**: 先预测 service，再根据预测的 service 选择对应的 user（需修改模型结构）
3. **分离指标**: 单独报告 service 准确率和 user 准确率
4. **多任务学习**: 在损失函数中加入更多约束（如正则化项）

## 测试建议

```bash
# 小数据集过拟合测试
python train.py --model hier --overfit-test --epochs 100 --batch-size 128

# 完整训练
python train.py --model hier --data data/train.npz --epochs 50 --batch-size 256 --lr 1e-2

# 与 baseline 对比
python train.py --model transformer --data data/train.npz --epochs 50
```
