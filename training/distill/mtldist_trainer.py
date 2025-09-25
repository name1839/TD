from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import DetectionModel, torch_safe_load
from ultralytics import YOLO
from copy import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.metrics import bbox_iou, probiou
from ultralytics.utils.tal import bbox2dist, dist2bbox, make_anchors, TaskAlignedAssigner
from ultralytics.utils.ops import xywh2xyxy
from triplet_mapper import get_triplet_mapper
from mtl_detect import MTLDetect
import math

class MTLDistillLoss:

    def __init__(self, model, tal_topk=10):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1] 
        
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_distill = nn.BCEWithLogitsLoss(reduction="none")
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
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, model=None):
        loss = torch.zeros(5, device=self.device) 
        
        if len(preds) == 5:  # val mode
            y, feats, I_scores, V_scores, T_scores = preds
        else:  # train mode
            feats, I_scores, V_scores, T_scores = preds
        
        reshaped_tensors = []
        for xi in feats:
            batch_size = feats[0].shape[0]
            reshaped_xi = xi.view(batch_size, self.no, -1)
            reshaped_tensors.append(reshaped_xi)

        concatenated = torch.cat(reshaped_tensors, dim=2)
        I_scores = torch.cat(I_scores, dim=2)
        V_scores = torch.cat(V_scores, dim=2)
        T_scores = torch.cat(T_scores, dim=2)

        pred_distri, pred_scores = concatenated.split([self.reg_max * 4, self.nc], dim=1)

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

        loss_main = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # triplet cls loss

        loss_bbox = torch.tensor(0.0, device=self.device)
        loss_dfl = torch.tensor(0.0, device=self.device)   
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss_bbox, loss_dfl = self.bbox_loss(
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
            loss_auxi = self.bce(pred_i_fg, i_targets).sum() / num_fg
            loss_auxv = self.bce(pred_v_fg, v_targets).sum() / num_fg
            loss_auxt = self.bce(pred_t_fg, t_targets).sum() / num_fg
        else:
            loss_auxi = torch.tensor(0.0, device=self.device)
            loss_auxv = torch.tensor(0.0, device=self.device)
            loss_auxt = torch.tensor(0.0, device=self.device)

        if hasattr(model, 'teacher') and model.teacher is not None:
            with torch.no_grad():
                teacher_preds = model.teacher(batch['img'])
            
            if len(teacher_preds) == 4:  # train
                teacher_feats, teacher_I, teacher_V, teacher_T = teacher_preds
                
                teacher_concat = torch.cat([xi.view(teacher_feats[0].shape[0], self.no, -1) for xi in teacher_feats], 2)
                teacher_distri, teacher_scores = teacher_concat.split([self.reg_max * 4, self.nc], 1)
                teacher_scores = teacher_scores.permute(0, 2, 1).contiguous()
                
                teacher_I_concat = torch.cat(teacher_I, dim=2).permute(0, 2, 1).contiguous()
                teacher_V_concat = torch.cat(teacher_V, dim=2).permute(0, 2, 1).contiguous()
                teacher_T_concat = torch.cat(teacher_T, dim=2).permute(0, 2, 1).contiguous()
                
            else:  # val
                teacher_output = teacher_preds[0]
                teacher_scores = teacher_output[:, -self.nc:, :].permute(0, 2, 1).contiguous()
                teacher_I_concat = teacher_V_concat = teacher_T_concat = None

            if fg_mask.sum() > 0:
                fg_indices = fg_mask.view(-1)
                teacher_triplet_fg = teacher_scores.view(-1, teacher_scores.shape[-1])[fg_indices]
                pred_triplet_fg = pred_scores.view(-1, pred_scores.shape[-1])[fg_indices]
                target_triplet_fg = target_scores.view(-1, target_scores.shape[-1])[fg_indices]
                
                loss[1] = self.compute_distill_loss(teacher_triplet_fg, pred_triplet_fg, target_triplet_fg)
            else:
                loss[1] = torch.tensor(0.0, device=self.device)
            
            if fg_mask.sum() > 0 and teacher_I_concat is not None:
                teacher_i_fg = teacher_I_concat.view(-1, self.nc_instrument)[fg_indices]
                teacher_v_fg = teacher_V_concat.view(-1, self.nc_action)[fg_indices]
                teacher_t_fg = teacher_T_concat.view(-1, self.nc_target)[fg_indices]
                
                distill_loss_i = self.compute_distill_loss(teacher_i_fg, pred_i_fg, i_targets)
                distill_loss_v = self.compute_distill_loss(teacher_v_fg, pred_v_fg, v_targets)
                distill_loss_t = self.compute_distill_loss(teacher_t_fg, pred_t_fg, t_targets)
                
                loss[2] = distill_loss_i
                loss[3] = distill_loss_v
                loss[4] = distill_loss_t

        loss[0] = loss_main + loss_bbox + loss_dfl + loss_auxi + loss_auxv + loss_auxt
        loss[0] *= 0   
        loss[1] *= 1   
        loss[2] *= 0.5   
        loss[3] *= 0.1   
        loss[4] *= 0.1   

        return loss * batch_size, loss.detach()

 
    # def compute_auxiliary_distill_loss(self, teacher_aux, student_aux, component_targets, fg_mask,
    #                                  tweight=0.7, temperature=3.0):
    #     """Compute auxiliary component distillation loss (I/V/T) - targets already filtered."""
    #     if fg_mask.sum() == 0:
    #         return torch.tensor(0.0, device=self.device)
        
    #     # Apply foreground mask to teacher and student predictions
    #     fg_indices = fg_mask.view(-1)
    #     teacher_fg = teacher_aux.view(-1, teacher_aux.shape[-1])[fg_indices]
    #     student_fg = student_aux.view(-1, student_aux.shape[-1])[fg_indices]
        
    #     # component_targets is already filtered to foreground only
    #     # Soft teacher targets with temperature scaling
    #     teacher_soft = (teacher_fg / temperature).sigmoid()
        
    #     # Mix teacher soft targets with hard GT targets
    #     mixed_targets = teacher_soft * tweight + component_targets * (1 - tweight)
        
    #     # BCE loss with mixed targets
    #     loss_aux = self.bce_distill(student_fg, mixed_targets.to(student_fg.dtype)).sum() / fg_mask.sum()
        
    #     return loss_aux
    
    def compute_distill_loss(self, teacher_pred, student_pred, gt_targets, 
                                   tweight=0.7, temperature=3.0):
        if teacher_pred.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)
        
        teacher_soft = (teacher_pred / temperature).sigmoid()
        
        mixed_targets = teacher_soft * tweight + gt_targets * (1 - tweight)
        
        loss_distill = self.bce_distill(student_pred, mixed_targets.to(student_pred.dtype)).sum() / teacher_pred.shape[0]
        
        return loss_distill


class MTLDistillModel(DetectionModel):
    def __init__(self, cfg='yolo12m-mtl.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        self.teacher = None

    def init_criterion(self):
        return MTLDistillLoss(self)
    
    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch, model=self)


class MTLDistillTrainer(DetectionTrainer):  
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        self.teacher_path = None
        if overrides and 'teacher_path' in overrides:
            overrides = overrides.copy()
            self.teacher_path = overrides.pop('teacher_path')

        if cfg is None:
            import sys
            sys.path.insert(0, 'lib')
            from ultralytics.utils import DEFAULT_CFG_DICT
            cfg = DEFAULT_CFG_DICT
        super().__init__(cfg, overrides, _callbacks)
        
        self.teacher = None
        if self.teacher_path:
            self._setup_teacher()
    
    def _setup_teacher(self):       
        self.teacher = MTLDistillModel('yolo12m-mtl.yaml', nc=self.data["nc"], verbose=False)

        ckpt, _ = torch_safe_load(self.teacher_path)
        if ckpt.get("ema") is not None:
            self.teacher.load(ckpt["ema"])
        elif ckpt.get("model") is not None:
            self.teacher.load(ckpt["model"])
        else:
            raise ValueError(f"No valid model weights found in {self.teacher_path}")
        
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def get_model(self, cfg=None, weights=None, verbose=True):
        if cfg is None:
            cfg = "yolo12m-mtl.yaml"
        
        model = MTLDistillModel(cfg, nc=self.data["nc"], verbose=verbose)
        
        if weights:
            model.load(weights)
        elif hasattr(self.args, 'pretrained') and self.args.pretrained:
            pretrained_weights = self.args.pretrained
            ckpt, _ = torch_safe_load(pretrained_weights)
            model.load(ckpt)
        else:
            assert False, "No pretrained weights or trained weights found"
        
        self._reinit_auxiliary_heads(model)
        
        model.teacher = self.teacher
        
        self._apply_freeze_strategy(model)
        
        return model
    
    def _reinit_auxiliary_heads(self, model):
        detection_head = model.model[-1]
        reinit_count = 0
        
        if hasattr(detection_head, 'cv3'):
            for m in detection_head.cv3.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    reinit_count += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
        
        for aux_head in ['cv4', 'cv5', 'cv6']:
            if hasattr(detection_head, aux_head):
                aux_module = getattr(detection_head, aux_head)
                for m in aux_module.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                        reinit_count += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)      
        print(f"   Total reinitialized parameters: {reinit_count}")
    
    def _apply_freeze_strategy(self, model):       
        for i in range(21):
            for param in model.model[i].parameters():
                param.requires_grad = False
        
        detection_head = model.model[-1]
        if hasattr(detection_head, 'cv2'):
            for param in detection_head.cv2.parameters():
                param.requires_grad = False
        if hasattr(detection_head, 'dfl'):
            for param in detection_head.dfl.parameters():
                param.requires_grad = False
        
        if hasattr(detection_head, 'cv3'):
            for param in detection_head.cv3.parameters():
                param.requires_grad = True
        
        for aux_head in ['cv4', 'cv5', 'cv6']:
            if hasattr(detection_head, aux_head):
                for param in getattr(detection_head, aux_head).parameters():
                    param.requires_grad = True
        
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        total_count = sum(1 for p in model.parameters())
        print(f"   Trainable: {trainable_count}/{total_count} parameters (CV3 + auxiliary heads)")
    
    def get_validator(self):
        self.loss_names = ("ot_loss", "tri_loss", "i_loss", "a_loss", "t_loss")
        return DetectionValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args), 
            _callbacks=self.callbacks
        )

class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max=16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
