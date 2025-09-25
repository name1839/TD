from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics import YOLO
from copy import copy
import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.block import DFL
from mtl_detect import MTLDetect
from ultralytics.nn.tasks import DetectionModel, torch_safe_load
from ultralytics.utils.metrics import bbox_iou, probiou
from ultralytics.utils.tal import bbox2dist, dist2bbox, make_anchors, TaskAlignedAssigner
from ultralytics.utils.ops import xywh2xyxy
import torch.nn.functional as F
from triplet_mapper import get_triplet_mapper
from ultralytics.models.yolo.detect import DetectionValidator

class MTLTrainer(DetectionTrainer):
  
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        if cfg is None:
            import sys
            sys.path.insert(0, 'lib')
            from ultralytics.utils import DEFAULT_CFG_DICT
            cfg = DEFAULT_CFG_DICT
        super().__init__(cfg, overrides, _callbacks)
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        if cfg is None:
            cfg = "yolo12m-mtl.yaml"
        
        model = MTLModel(cfg, nc=self.data["nc"], verbose=verbose)

        if weights:
            model.load(weights)
        elif hasattr(self.args, 'pretrained') and self.args.pretrained:
            pretrained_weights = self.args.pretrained
            ckpt, _ = torch_safe_load(pretrained_weights)
            model.load(ckpt)
        else:
            assert False, "No pretrained weights or trained weights found"
        return model
    
    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "i_loss", "v_loss", "t_loss")
        return DetectionValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args), 
            _callbacks=self.callbacks
        )

class MTLModel(DetectionModel):

    def __init__(self, cfg='yolo12m-mtl.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)

        self.teacher = None

    def init_criterion(self):
        return MTLLoss(self)
    
    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds #[1,71,2100], [1,89,2100]
        return self.criterion(preds, batch, model=None)

class MTLLoss:
    def __init__(self, model, tal_topk=10): 
        device = next(model.parameters()).device  
        h = model.args 

        m = model.model[-1]  
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride 
        self.nc = m.nc  
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.triplet_mapper = get_triplet_mapper()
        self.nc_instrument, self.nc_action, self.nc_target = self.triplet_mapper.get_component_counts()

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, model=None):
        loss = torch.zeros(6, device=self.device)  
        if len(preds) == 5: # val
            y, feats, I_scores, V_scores, T_scores = preds
        else: # train 
            feats, I_scores, V_scores, T_scores = preds
        
        reshaped_tensors = []
        for xi in feats: # P3, P4, P5
            batch_size = feats[0].shape[0]
            reshaped_xi = xi.view(batch_size, self.no, -1)
            reshaped_tensors.append(reshaped_xi)

        concatenated = torch.cat(reshaped_tensors, dim=2)
        I_scores = torch.cat(I_scores, dim=2)
        V_scores = torch.cat(V_scores, dim=2)
        T_scores = torch.cat(T_scores, dim=2)

        pred_distri, pred_scores = concatenated.split(
            split_size=[self.reg_max * 4, self.nc],
            dim=1
        )

        I_scores = I_scores.permute(0, 2, 1).contiguous()
        V_scores = V_scores.permute(0, 2, 1).contiguous()
        T_scores = T_scores.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  
        
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        if fg_mask.sum() > 0:
            fg_triplet_labels = target_labels[fg_mask]
            
            i_labels, v_labels, t_labels = self.triplet_mapper.triplet_to_components(fg_triplet_labels)
            
            i_targets = F.one_hot(i_labels, self.nc_instrument).float() 
            v_targets = F.one_hot(v_labels, self.nc_action).float()
            t_targets = F.one_hot(t_labels, self.nc_target).float()
            
            fg_indices = fg_mask.view(-1)  
            pred_i_fg = I_scores.view(-1, self.nc_instrument)[fg_indices]  
            pred_v_fg = V_scores.view(-1, self.nc_action)[fg_indices]    
            pred_t_fg = T_scores.view(-1, self.nc_target)[fg_indices] 
            
            num_fg = fg_mask.sum()
            loss[3] = self.bce(pred_i_fg, i_targets).sum() / num_fg
            loss[4] = self.bce(pred_v_fg, v_targets).sum() / num_fg
            loss[5] = self.bce(pred_t_fg, t_targets).sum() / num_fg
        else:
            loss[3] = torch.tensor(0.0, device=self.device)
            loss[4] = torch.tensor(0.0, device=self.device) 
            loss[5] = torch.tensor(0.0, device=self.device)
            
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls  
        loss[2] *= self.hyp.dfl  
        loss[3] *= self.hyp.cls * 0.5  
        loss[4] *= self.hyp.cls * 0.5  
        loss[5] *= self.hyp.cls * 0.5  

        return loss * batch_size, loss.detach()  

class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)
    
class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
