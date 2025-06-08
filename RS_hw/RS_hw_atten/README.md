# 图增强VAE推荐系统

这是一个基于图增强变分自编码器（Graph-Enhanced VAE）的推荐系统实现，用于预测用户对物品的评分。

## 模型特点

1. **单塔框架**：使用LightGCN处理用户-物品交互图
2. **图增强VAE**：对节点表征做分布式编码和重构，用于降噪和正则化
3. **层级注意力**：对不同层GCN的输出进行加权融合
4. **双重采样策略**：通过Gumbel-Softmax和Bernoulli采样稳定图结构
5. **端到端联合优化**：同时优化预测和VAE重构损失

## 数据格式

- **train.txt**：训练数据，格式为用户ID|物品数量，后跟物品ID和评分
- **test.txt**：测试数据，格式为用户ID|物品数量，后跟物品ID
- **result.txt**：结果文件，格式与train.txt相同

## 环境配置

需要的主要依赖包：

```
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
torch>=1.9.0
dgl>=0.7.0
tqdm>=4.62.0
```

可以通过以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python run.py train --data_path ./data --embedding_dim 64 --n_layers 3 --predict
```

主要参数：
- `--data_path`：数据目录
- `--embedding_dim`：嵌入维度
- `--n_layers`：GCN层数
- `--lr`：学习率
- `--vae_weight`：VAE损失权重
- `--batch_size`：批大小
- `--epochs`：训练轮数
- `--train_ratio`：训练集比例
- `--predict`：添加此参数将在训练后直接进行预测

### 只进行预测

```bash
python run.py predict --data_path ./data --model_path ./best_model.pt --result_path ./result.txt
```

主要参数：
- `--data_path`：数据目录
- `--model_path`：已训练模型的路径
- `--result_path`：结果保存路径

## 文件说明

- `model.py`：模型定义，包括LightGCN、VAE和层级注意力等组件
- `data_processor.py`：数据处理，负责加载和预处理数据
- `train.py`：训练脚本，包含训练和评估流程
- `predict.py`：预测脚本，用于加载已训练模型并进行预测
- `run.py`：主运行脚本，集成训练和预测功能

## 评估指标

使用均方根误差（RMSE）作为主要评估指标：

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

其中 $y_i$ 是真实评分，$\hat{y}_i$ 是预测评分。
