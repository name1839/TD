import torch
import numpy as np
from pathlib import Path


class TripletMapper:
    
    def __init__(self, mapping_file="triplet_maps_v2.txt"):
        self.mapping_file = Path(mapping_file)
        self.triplet_to_ivt = {}  
        self.num_instruments = 0
        self.num_actions = 0
        self.num_targets = 0
        
        self._load_mapping()
    
    def _load_mapping(self):
        if not self.mapping_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_file}")
        
        with open(self.mapping_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',')
            if len(parts) >= 4:
                triplet_id = int(parts[0])
                instrument_id = int(parts[1])
                action_id = int(parts[2])
                target_id = int(parts[3])
                
                self.triplet_to_ivt[triplet_id] = (instrument_id, action_id, target_id)
                
                self.num_instruments = max(self.num_instruments, instrument_id + 1)
                self.num_actions = max(self.num_actions, action_id + 1)
                self.num_targets = max(self.num_targets, target_id + 1)
        
    def get_component_counts(self):
        return self.num_instruments, self.num_actions, self.num_targets
    
    def triplet_to_components(self, triplet_labels):
        if isinstance(triplet_labels, torch.Tensor):
            device = triplet_labels.device
            triplet_labels_np = triplet_labels.cpu().numpy()
            use_torch = True
        else:
            triplet_labels_np = np.array(triplet_labels)
            use_torch = False
        
        instrument_labels = []
        action_labels = []
        target_labels = []
        
        for triplet_id in triplet_labels_np.flatten():
            triplet_id = int(triplet_id)
            if triplet_id in self.triplet_to_ivt:
                i_id, v_id, t_id = self.triplet_to_ivt[triplet_id]
                instrument_labels.append(i_id)
                action_labels.append(v_id)
                target_labels.append(t_id)
            else:
                print(f"Warning: Unknown triplet ID {triplet_id}, using default (0,0,0)")
                instrument_labels.append(0)
                action_labels.append(0)
                target_labels.append(0)
        
        # Convert back to original shape
        original_shape = triplet_labels_np.shape
        instrument_labels = np.array(instrument_labels).reshape(original_shape)
        action_labels = np.array(action_labels).reshape(original_shape)
        target_labels = np.array(target_labels).reshape(original_shape)
        
        if use_torch:
            instrument_labels = torch.from_numpy(instrument_labels).to(device)
            action_labels = torch.from_numpy(action_labels).to(device)
            target_labels = torch.from_numpy(target_labels).to(device)
        
        return instrument_labels, action_labels, target_labels
    
    def create_component_targets(self, input_data, num_classes_per_component=None):
        if num_classes_per_component is None:
            num_instruments, num_actions, num_targets = self.get_component_counts()
        else:
            num_instruments, num_actions, num_targets = num_classes_per_component
        
        device = input_data.device
        
        if input_data.dim() == 1:
            triplet_labels = input_data
            batch_size = triplet_labels.shape[0]
            
            instrument_labels, action_labels, target_labels = self.triplet_to_components(triplet_labels)
            
            instrument_targets = torch.zeros(batch_size, num_instruments, device=device)
            action_targets = torch.zeros(batch_size, num_actions, device=device)
            target_targets = torch.zeros(batch_size, num_targets, device=device)
            
            for i in range(batch_size):
                if instrument_labels[i] < num_instruments:
                    instrument_targets[i, instrument_labels[i]] = 1.0
                if action_labels[i] < num_actions:
                    action_targets[i, action_labels[i]] = 1.0
                if target_labels[i] < num_targets:
                    target_targets[i, target_labels[i]] = 1.0
                    
        elif input_data.dim() == 3:
            target_scores = input_data
            batch_size, num_anchors, num_triplet_classes = target_scores.shape
            
            triplet_labels = target_scores.argmax(dim=2) 
            
            triplet_labels_flat = triplet_labels.view(-1) 
            instrument_labels, action_labels, target_labels = self.triplet_to_components(triplet_labels_flat)
            
            instrument_labels = instrument_labels.view(batch_size, num_anchors)
            action_labels = action_labels.view(batch_size, num_anchors)
            target_labels = target_labels.view(batch_size, num_anchors)
            
            instrument_targets = torch.zeros(batch_size, num_anchors, num_instruments, device=device)
            action_targets = torch.zeros(batch_size, num_anchors, num_actions, device=device)
            target_targets = torch.zeros(batch_size, num_anchors, num_targets, device=device)
            
            instrument_targets.scatter_(2, instrument_labels.unsqueeze(-1), 1)
            action_targets.scatter_(2, action_labels.unsqueeze(-1), 1)
            target_targets.scatter_(2, target_labels.unsqueeze(-1), 1)
            
            triplet_mask = target_scores.sum(dim=2, keepdim=True) > 0 
            
            instrument_targets = instrument_targets * triplet_mask
            action_targets = action_targets * triplet_mask  
            target_targets = target_targets * triplet_mask
            
        else:
            raise ValueError(f"Unsupported input format. Expected 1D or 3D tensor, got {input_data.dim()}D tensor with shape {input_data.shape}")
        
        return instrument_targets, action_targets, target_targets

_global_mapper = None

def get_triplet_mapper(mapping_file="triplet_maps_v2.txt"):
    global _global_mapper
    if _global_mapper is None:
        _global_mapper = TripletMapper(mapping_file)
    return _global_mapper

def triplet_to_components(triplet_labels, mapping_file="triplet_maps_v2.txt"):

    mapper = get_triplet_mapper(mapping_file)
    return mapper.triplet_to_components(triplet_labels)

def create_component_targets(gt_labels, mapping_file="triplet_maps_v2.txt"):

    mapper = get_triplet_mapper(mapping_file)
    return mapper.create_component_targets(gt_labels)