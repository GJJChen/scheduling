
# 单AP多用户多业务调度 —— 规则学习 (BiLSTM)
本项目包含两部分：
1) **数据集生成**：随机生成 1ms 的系统状态快照 `[N_user, 3, 2]`，其中 3 个业务分别为 `VO, VI, BE`，2 个属性为 `[buffer_bytes, wait_ms]`。标签是 **传统调度规则** 在该快照下应当调度的**用户ID**。
2) **监督训练**：使用 BiLSTM（可切换为 MLP/Transformer）学习“传统调度规则”，并报告 **AI调度 vs 传统调度** 的匹配准确率。

> 说明：为便于在“**没有时间连续性**”的前提下刻画 BE 最小带宽保障（“每 100ms 至少一次发送机会”），本实现约定 **BE 的 `wait_ms` 即为“距上次获得 BE 发送机会的时间”**。因此，当 `BE.wait_ms >= 100` 且 `buffer_bytes > 0` 时，认为存在最小带宽需求。

## 目录
```
scheduler_ai/
  ├── config.py
  ├── rules.py
  ├── generate_dataset.py
  ├── dataset.py
  ├── model.py
  ├── train.py
  ├── requirements.txt
  ├── data/
  └── checkpoints/
```

## 快速开始
```bash
# 1) 创建并激活虚拟环境（可选）
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) 安装依赖
pip install -r requirements.txt

# 3) 生成数据集（默认 200k 样本）
python generate_dataset.py --n 200000 --out data/train.npz

# 4) 训练（bilstm / mlp / transformer 均可）
python train.py --data data/train.npz --model bilstm --epochs 5 --batch-size 256

# 5) 评估输出会显示训练/验证准确率与最佳模型保存路径
```

## 调度规则回顾（用于打标签）
- **物理层**：双流 MCS11 速率 **2442 Mbps**；**EDCA+Preamble 固定开销 150 µs**；**单次最大发送时长 4 ms**。发送时长：`min( 0.150 ms + payload_bits / 2442Mbps, 4 ms )`。
- **Step 1（VO）**：若存在 `VO.buffer>0`，**严格优先级** —— 选取 **wait_ms 最大** 的 VO。若某些 VO 的 `wait_ms>20ms`，**丢弃并记负面事件**（不参与调度）。若全部 VO 都过期被丢弃，则进入 Step 2。
- **Step 2（VI）**：若存在 `VI.buffer>0` 且 `wait_ms>20ms`，**严格优先级** —— 选取 **wait_ms 最大** 的 VI；其中 `wait_ms>50ms` 记**负面事件**（仍发送，不丢弃）。若所有 VI 都在 20ms 以内，则进入 Step 3。
- **Step 3（BE 最小保障）**：若存在 `BE.buffer>0` 且 `wait_ms>=100ms`（表示最小带宽需求触发），选取 **wait_ms 最大** 的 BE。否则进入 Step 4。
- **Step 4（时长最长）**：在剩余的 VI/BE 候选中（`buffer>0`），按**单次发送时长**从大到小选择**时长最长**的一个。若系统无可发送缓存，则空闲 1ms。

> 生成数据时确保“至少一个可调度候选”，因此不会出现“空闲标签”。

---

## 备注
- 用户数 N 固定为 **128**（与需求一致）。
- 输入特征为每用户的 3×2=6 维（`VO/VI/BE × (buffer_bytes, wait_ms)`）。模型按 **用户维度**（长度 128）做序列建模并输出“应调度的用户ID”。
- 你可以在 `config.py` 里调整阈值、速率、采样分布等。
