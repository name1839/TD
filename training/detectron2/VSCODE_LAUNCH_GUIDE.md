# 🚀 VSCode Launch Configurations Guide

## 📋 如何使用

1. **打开VSCode**，确保在detectron2项目根目录
2. **按F5**或点击"Run and Debug"
3. **选择你想要的配置**
4. **点击运行**

## 🎯 配置分类

### 🚀 快速测试 (1-5分钟)
- `🚀 Quick Test - Faster R-CNN Split1 (100 iter)` - **默认启用**
- `🚀 Quick Test - RetinaNet Split1 (100 iter)` - 取消注释使用

### 🏃 Faster R-CNN 训练
- `🏃 Train Faster R-CNN - Split1/2/3/4/5` - 所有split的Faster R-CNN训练

### 🎯 RetinaNet 训练  
- `🎯 Train RetinaNet - Split1/2` - RetinaNet训练（可扩展到其他split）

### 🏆 Cascade R-CNN 训练
- `🏆 Train Cascade R-CNN - Split1/2` - 高精度Cascade R-CNN训练

### 🌟 Swin Transformer 训练
- `🌟 Train SwinT - Split1/2` - 最先进的SwinT训练（需要先运行setup）

### 🔍 推理/评估
- `🔍 Inference - [Model] Split1 (Test Set)` - 在测试集上评估训练好的模型

### 🚀 批量实验
- `🚀 Batch - All Models on Split1` - 在split1上训练所有模型
- `🚀 Batch - Faster R-CNN on All Splits` - 在所有split上训练Faster R-CNN

### ⚙️ 设置工具
- `⚙️ Setup SwinT Integration` - 设置SwinT集成
- `⚙️ Register Triplet Datasets` - 注册数据集
- `⚙️ Generate Split Configs` - 生成split配置文件

### 🔧 自定义训练
- `🔧 Custom - Fast Training (40k iter)` - 快速训练（较少迭代）
- `🔧 Custom - High LR Training` - 高学习率训练
- `🔧 Custom - Small Batch Training` - 小批量训练（节省GPU内存）

### 🔄 恢复训练
- `🔄 Resume - Faster R-CNN Split1` - 从检查点恢复训练

## 📝 使用步骤

### 1. 首次使用
```
1. 选择 "🚀 Quick Test - Faster R-CNN Split1 (100 iter)"
2. 按F5运行
3. 等待1-2分钟验证系统正常
```

### 2. 正式训练
```
1. 取消注释你想要的训练配置
2. 注释掉当前启用的配置
3. 按F5运行
```

### 3. 推理评估
```
1. 确保模型已训练完成（存在model_final.pth）
2. 取消注释对应的推理配置
3. 按F5运行
```

## 🔧 自定义配置

### 修改参数
在`args`数组中添加或修改参数：
```json
"args": [
    "--config-file", "configs/Triplet-Detection/splits/faster_rcnn_split1.yaml",
    "--num-gpus", "1",
    "SOLVER.BASE_LR", "0.02",           // 自定义学习率
    "SOLVER.MAX_ITER", "120000",        // 自定义迭代次数
    "OUTPUT_DIR", "./output/my_experiment"  // 自定义输出目录
]
```

### 添加新配置
复制现有配置并修改：
```json
{
    "name": "🔧 My Custom Training",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/tools/train_triplet.py",
    "args": [
        // 你的自定义参数
    ],
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}"
}
```

## 💡 使用技巧

### 快速切换配置
1. **注释当前配置**：在配置前后加上`//`
2. **取消注释目标配置**：删除`//`
3. **保存文件**：Ctrl+S
4. **运行**：F5

### 监控训练
- 训练日志会在VSCode的集成终端中显示
- 输出文件保存在`OUTPUT_DIR`指定的目录
- 可以用TensorBoard查看训练曲线

### 调试模式
- 在代码中设置断点
- 使用"🐛 Debug - Current File"配置调试单个文件
- 可以逐步调试训练过程

## 🎯 推荐工作流

### 初学者
1. `🚀 Quick Test` → 验证系统
2. `🏃 Train Faster R-CNN - Split1` → 基线模型
3. `🔍 Inference - Faster R-CNN Split1` → 评估结果

### 进阶用户
1. `🚀 Batch - All Models on Split1` → 模型比较
2. `🌟 Train SwinT` → 最佳性能
3. `🔧 Custom Training` → 参数调优

### 研究用户
1. `🚀 Batch - Faster R-CNN on All Splits` → 交叉验证
2. 自定义配置 → 实验设计
3. 批量推理 → 结果分析

## 🚨 注意事项

- **确保conda环境**：运行前激活detectron2环境
- **GPU内存**：如果内存不足，使用小批量配置
- **路径检查**：确保模型权重路径正确
- **数据集注册**：首次使用前运行数据集注册

现在你可以轻松地在VSCode中运行所有训练和推理任务！🎉
