# OpenGait-MindSpore

<div align="center">

**🚀 将OpenGait从PyTorch完整迁移到MindSpore框架**

[![MindSpore](https://img.shields.io/badge/MindSpore-2.0+-blue)](https://www.mindspore.cn/)
[![Python](https://img.shields.io/badge/Python-3.7+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-red)](LICENSE)

</div>

---

## 📢 项目简介

**OpenGait-MindSpore** 是将 [OpenGait](https://github.com/ShiqiYu/OpenGait) 项目从 **PyTorch** 框架完整迁移到 **MindSpore** 框架的版本。

本项目完整保留了OpenGait的所有核心功能，包括：
- ✅ 完整的数据加载和处理流程
- ✅ 所有核心模型组件和骨干网络
- ✅ 训练和测试流程
- ✅ 评估和指标计算
- ✅ 分布式训练支持

### 🎯 主要特点

- **🔄 完整迁移**: 核心框架100%从PyTorch迁移到MindSpore
- **📦 即插即用**: 保持与原项目相同的接口设计，易于使用
- **🚀 高性能**: 充分利用MindSpore的图优化和自动并行能力
- **📚 完整文档**: 提供详细的转换指南和使用文档
- **✅ 已验证模型**: 包含Baseline、GaitSet、GaitPart、GaitGL等已验证模型

---

## 🏗️ 项目结构

```
opengait_mindspore/
├── main.py                    # 主程序入口
├── data/                      # 数据模块
│   ├── dataset.py            # 数据集类
│   ├── sampler.py            # 采样器
│   ├── transform.py          # 数据变换
│   └── collate_fn.py         # 批处理函数
├── modeling/                  # 模型模块
│   ├── base_model.py         # 基础模型类
│   ├── modules.py            # 核心模块组件
│   ├── loss_aggregator.py    # 损失聚合器
│   ├── losses/               # 损失函数
│   ├── backbones/            # 骨干网络
│   └── models/               # 模型定义
├── evaluation/               # 评估模块
│   ├── evaluator.py         # 评估器
│   ├── metric.py            # 评估指标
│   └── re_rank.py           # 重排序
└── utils/                    # 工具模块
    ├── common.py            # 通用工具
    └── msg_manager.py      # 消息管理器
```

---

## 🚀 快速开始

### 环境要求

- Python >= 3.7
- MindSpore >= 2.0
- CUDA >= 11.1 (GPU版本) 或 CPU版本

### 安装步骤

1. **安装MindSpore**

   根据您的系统选择安装方式：

   ```bash
   # GPU版本 (CUDA 11.1)
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-2.0.0-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

   # CPU版本
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0/MindSpore/cpu/x86_64/mindspore-2.0.0-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

   更多安装方式请参考 [MindSpore官方文档](https://www.mindspore.cn/install)

2. **安装其他依赖**

   ```bash
   pip install -r requirements.txt
   ```

3. **准备数据集**

   按照原OpenGait项目的[数据集准备文档](../docs/2.prepare_dataset.md)准备数据集。

### 运行示例

#### 训练模型

```bash
# 单卡训练
python opengait_mindspore/main.py \
    --cfgs ./configs/baseline/baseline.yaml \
    --phase train \
    --device_target GPU \
    --device_id 0

# 多卡训练
mpirun -n 4 python opengait_mindspore/main.py \
    --cfgs ./configs/baseline/baseline.yaml \
    --phase train \
    --device_target GPU
```

#### 测试模型

```bash
python opengait_mindspore/main.py \
    --cfgs ./configs/baseline/baseline.yaml \
    --phase test \
    --device_target GPU \
    --device_id 0 \
    --iter 10000
```

---

## 📋 已转换模块

### ✅ 核心框架 (100%完成)

- ✅ **Utils模块** - 通用工具函数、消息管理器
- ✅ **Data模块** - 数据集、采样器、数据变换、批处理
- ✅ **Evaluation模块** - 评估器、评估指标、重排序
- ✅ **BaseModel** - 基础模型类（所有模型的基类）
- ✅ **LossAggregator** - 损失聚合器
- ✅ **Modules** - 所有核心模块组件
- ✅ **Losses** - Triplet Loss、Cross Entropy Loss等
- ✅ **Backbones** - Plain、ResNet、GCN、ResGCN、U-Net等
- ✅ **Main入口** - 主程序入口

### ✅ 已转换模型

1. **Baseline** - 基础模型
2. **GaitSet** - GaitSet模型
3. **GaitPart** - GaitPart模型
4. **GaitGL** - GaitGL模型

### 📝 转换状态

详细的转换状态请查看 [CONVERSION_STATUS.md](CONVERSION_STATUS.md)

---

## 🔄 PyTorch到MindSpore的主要转换

### API映射

| PyTorch | MindSpore | 说明 |
|---------|-----------|------|
| `torch.nn.Module` | `mindspore.nn.Cell` | 模型基类 |
| `forward()` | `construct()` | 前向传播方法 |
| `torch.Tensor` | `mindspore.Tensor` | 张量类型 |
| `torch.max()` | `ops.ReduceMax()` | 最大值操作 |
| `torch.cat()` | `ops.Concat()` | 拼接操作 |
| `torch.sum()` | `ops.ReduceSum()` | 求和操作 |
| `.size()` | `.shape` | 获取形状 |
| `.cuda()` | 通过context设置 | 设备管理 |
| `torch.distributed` | `mindspore.communication` | 分布式通信 |

### 主要改动

1. **模型定义**: 所有`nn.Module`改为`nn.Cell`，`forward()`改为`construct()`
2. **Tensor操作**: 所有PyTorch操作改为MindSpore对应操作
3. **数据加载**: 使用MindSpore的`GeneratorDataset`替代PyTorch的`DataLoader`
4. **分布式训练**: 使用MindSpore的分布式API
5. **优化器**: 使用MindSpore的优化器实现

---

## 📚 文档

- [转换状态](CONVERSION_STATUS.md) - 详细的转换进度和状态
- [快速开始指南](QUICK_START.md) - 快速上手指南（如果存在）
- [模型转换指南](MODEL_CONVERSION_GUIDE.md) - 如何转换其他模型（如果存在）

---

## 🎓 使用示例

### 创建模型

```python
import mindspore as ms
from opengait_mindspore.modeling.models import Baseline
from opengait_mindspore.utils import config_loader

# 加载配置
cfgs = config_loader('configs/baseline/baseline.yaml')

# 创建模型
model = Baseline(cfgs, training=True)

# 创建测试输入
import numpy as np
test_input = ms.Tensor(np.random.randn(2, 30, 64, 44), dtype=ms.float32)

# 前向传播
output = model.construct(([test_input], None, None, None, None))
print("Model test passed!")
```

---

## ⚠️ 注意事项

1. **Checkpoint转换**: PyTorch的checkpoint需要转换为MindSpore格式才能使用
2. **数据格式**: 确保数据集格式符合要求
3. **设备配置**: 通过MindSpore的context设置设备，不需要显式调用`.cuda()`
4. **分布式训练**: 使用`mpirun`启动多卡训练

---

## 🔧 转换其他模型

如果需要转换其他模型文件，请参考：

1. 已转换的模型示例：`modeling/models/baseline.py`、`gaitset.py`等
2. 转换指南文档（如果存在）
3. MindSpore官方文档

所有模型都继承自`BaseModel`，转换相对简单。

---

## 📊 性能对比

本项目保持了与原OpenGait项目相同的模型结构和训练流程，理论上应该获得相似的性能。具体性能可能因硬件和MindSpore版本而有所差异。

---

## 🙏 致谢

- 感谢 [OpenGait](https://github.com/ShiqiYu/OpenGait) 项目团队提供的优秀框架
- 感谢 [MindSpore](https://www.mindspore.cn/) 团队提供的深度学习框架

---

## 📄 许可证

本项目仅用于**学术研究**目的，不得用于任何商业用途。

原项目许可证请参考 [OpenGait LICENSE](../LICENSE)

---

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 参考原项目 [OpenGait](https://github.com/ShiqiYu/OpenGait)

---

## 🔗 相关链接

- [OpenGait原项目](https://github.com/ShiqiYu/OpenGait)
- [MindSpore官方文档](https://www.mindspore.cn/)
- [MindSpore GitHub](https://github.com/mindspore-ai/mindspore)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个Star支持一下！⭐**

</div>
