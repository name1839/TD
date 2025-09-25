# 🎯 Triplet Detection System - Complete Usage Guide

## 📋 Available Models

### Detectron2-based Models
- **faster_rcnn** - Faster R-CNN with ResNet-50 FPN (recommended baseline)
- **retinanet** - RetinaNet with ResNet-50 FPN (single-stage detector)
- **cascade_rcnn** - Cascade R-CNN with ResNet-50 FPN (higher accuracy)
- **vitdet** - ViT-Det (simplified version, experimental)

## 🚀 Quick Start Commands

### (1) Test Setup (Fast)
```bash
# Test with minimal iterations
python tools/train_triplet.py \
    --config-file configs/Triplet-Detection/splits/faster_rcnn_split1.yaml \
    --num-gpus 1 \
    SOLVER.MAX_ITER 1000 \
    OUTPUT_DIR ./output/test
```

### (2) Train on Specific Split
```bash
# Faster R-CNN on split2
python tools/train_triplet.py \
    --config-file configs/Triplet-Detection/splits/faster_rcnn_split2.yaml \
    --num-gpus 1

# RetinaNet on split3  
python tools/train_triplet.py \
    --config-file configs/Triplet-Detection/splits/retinanet_split3.yaml \
    --num-gpus 1

# Cascade R-CNN on split1
python tools/train_triplet.py \
    --config-file configs/Triplet-Detection/splits/cascade_rcnn_split1.yaml \
    --num-gpus 1
```

### (3) SwinT Training (After Setup)
```bash
# First setup SwinT
python setup_swint.py

# Then train
python tools/train_triplet.py \
    --config-file configs/Triplet-Detection/splits/swint_faster_rcnn_split1.yaml \
    --num-gpus 1
```

## 🔧 Advanced Usage

### Batch Experiments
```bash
# Train all models on split1
python tools/run_triplet_experiments.py --model all --split split1

# Train Faster R-CNN on all splits
python tools/run_triplet_experiments.py --model faster_rcnn --split all

# Train specific model on specific split
python tools/run_triplet_experiments.py --model cascade_rcnn --split split2
```

### Custom Parameters
```bash
# Adjust learning rate and iterations
python tools/train_triplet.py \
    --config-file configs/Triplet-Detection/splits/faster_rcnn_split1.yaml \
    SOLVER.BASE_LR 0.02 \
    SOLVER.MAX_ITER 120000 \
    OUTPUT_DIR ./output/custom_training
```

### Evaluation Only
```bash
python tools/train_triplet.py \
    --config-file configs/Triplet-Detection/splits/faster_rcnn_split1.yaml \
    --eval-only \
    MODEL.WEIGHTS ./output/faster_rcnn_split1/model_final.pth
```

## 📁 File Structure

```
detectron2/
├── configs/Triplet-Detection/
│   ├── Base-Triplet.yaml                    # Base configuration
│   ├── faster_rcnn_R_50_FPN_triplet.yaml    # Model configs
│   ├── retinanet_R_50_FPN_triplet.yaml
│   ├── cascade_rcnn_R_50_FPN_triplet.yaml
│   ├── vitdet_base_triplet.yaml
│   ├── swint_faster_rcnn_triplet.yaml
│   └── splits/                              # Split-specific configs
│       ├── faster_rcnn_split1.yaml
│       ├── faster_rcnn_split2.yaml
│       ├── ... (25 total configs)
│       └── README.md
├── tools/
│   ├── train_triplet.py                     # Main training script
│   ├── run_triplet_experiments.py           # Batch experiment runner
│   └── generate_split_configs.py            # Config generator
├── register_triplet.py                      # Dataset registration
└── setup_swint.py                          # SwinT integration
```

## 🎛️ Configuration Details

### Default Settings (Optimized for Surgical Tools)
- **Classes**: 89 triplet combinations (tool-action-target)
- **Learning Rate**: 0.01 (conservative for fine-grained detection)
- **Batch Size**: 8 (suitable for single GPU)
- **Max Iterations**: 80,000 (sufficient training)
- **Anchors**: Smaller sizes [16, 32, 64, 128, 256] for precise tools
- **Input Size**: Multi-scale training (480-640px)

### Model-Specific Optimizations
- **SwinT**: Lower LR (0.0001), higher weight decay (0.05)
- **Cascade R-CNN**: Class-agnostic bbox regression, multiple IoU thresholds
- **RetinaNet**: Focal loss, optimized anchor ratios

## 🔄 Workflow Recommendations

### 1. Initial Testing
```bash
# Quick test to verify setup
python tools/train_triplet.py \
    --config-file configs/Triplet-Detection/splits/faster_rcnn_split1.yaml \
    SOLVER.MAX_ITER 100 OUTPUT_DIR ./output/quick_test
```

### 2. Model Comparison
```bash
# Compare different models on same split
python tools/run_triplet_experiments.py --model faster_rcnn --split split1
python tools/run_triplet_experiments.py --model retinanet --split split1  
python tools/run_triplet_experiments.py --model cascade_rcnn --split split1
```

### 3. Cross-Validation
```bash
# Train best model on all splits
python tools/run_triplet_experiments.py --model faster_rcnn --split all
```

## 🚨 Troubleshooting

### Common Issues
1. **Dataset not registered**: Run `python register_triplet.py` first
2. **CUDA out of memory**: Reduce `SOLVER.IMS_PER_BATCH`
3. **SwinT import error**: Run `python setup_swint.py` first

### Performance Tips
- Use `--resume` to continue interrupted training
- Monitor GPU memory with smaller batch sizes if needed
- Use `DATALOADER.NUM_WORKERS 2` if I/O is slow

## 📊 Expected Results

Based on similar surgical tool detection tasks:
- **Faster R-CNN**: ~40-45 mAP
- **RetinaNet**: ~38-43 mAP  
- **Cascade R-CNN**: ~42-47 mAP
- **SwinT**: ~45-50 mAP (if properly configured)

Results will vary based on:
- Dataset quality and size
- Training duration
- Hyperparameter tuning
- Hardware specifications

## 🎯 Next Steps

1. **Start with Faster R-CNN on split1** for baseline
2. **Compare models** using the same split
3. **Fine-tune hyperparameters** based on initial results
4. **Scale to all splits** for cross-validation
5. **Consider ensemble methods** for final deployment
