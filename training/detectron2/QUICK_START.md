# 🚀 Triplet Detection - Quick Start Guide

## 📋 Available Models (All Detectron2-based)

- **faster_rcnn** - Faster R-CNN (recommended baseline)
- **retinanet** - RetinaNet (single-stage)
- **cascade_rcnn** - Cascade R-CNN (higher accuracy)
- **vitdet** - ViT-Det (experimental)

## ⚡ Quick Commands

### 1. Test Setup (1 minute)
```bash
conda activate detectron2
cd /ssd/prostate/detectron2

python tools/train_triplet.py \
    --config-file configs/Triplet-Detection/splits/faster_rcnn_split1.yaml \
    SOLVER.MAX_ITER 100 OUTPUT_DIR ./output/test
```

### 2. Train on Any Split
```bash
# Faster R-CNN on split2
python tools/train_triplet.py \
    --config-file configs/Triplet-Detection/splits/faster_rcnn_split2.yaml

# RetinaNet on split3
python tools/train_triplet.py \
    --config-file configs/Triplet-Detection/splits/retinanet_split3.yaml

# Cascade R-CNN on split1  
python tools/train_triplet.py \
    --config-file configs/Triplet-Detection/splits/cascade_rcnn_split1.yaml
```



### 3. Batch Experiments
```bash
# All models on split1
python tools/run_triplet_experiments.py --model all --split split1

# Faster R-CNN on all splits
python tools/run_triplet_experiments.py --model faster_rcnn --split all
```

## 📁 What's Available

✅ **20 Ready-to-use Configs**: 4 models × 5 splits
✅ **Automatic Dataset Registration**: No manual setup needed
✅ **Optimized for 89 Triplet Classes**: All parameters tuned
✅ **Multi-GPU Support**: Just change `--num-gpus`
✅ **Easy Split Switching**: One command per split

## 🎯 Recommended Workflow

1. **Quick test**: Run the 1-minute test above
2. **Baseline**: Train Faster R-CNN on split1
3. **Compare models**: Try different models on same split
4. **Scale up**: Use batch experiments for full evaluation

## 💡 Pro Tips

- Use `--resume` to continue interrupted training
- Reduce `SOLVER.IMS_PER_BATCH` if GPU memory is low
- Check `./output/[experiment]/` for results and logs
- All configs are pre-optimized for surgical tool detection

## 🔧 System Features

- **89 Triplet Classes**: Tool-Action-Target combinations
- **Multi-scale Training**: 480-640px input sizes
- **Conservative Learning**: 0.01 LR for stable training
- **Smart Anchors**: Smaller sizes for precise surgical tools
- **Auto Evaluation**: Every 5000 iterations

Ready to start training! 🎉
