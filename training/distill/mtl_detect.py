from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics import YOLO
import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.block import DFL

class MTLDetect(Detect):
    
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        
        self.nc_tool = 7
        self.nc_action = 10
        self.nc_target = 10
        c3 = max(ch[0], min(self.nc, 100)) 
        
        self.cv4 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc_tool, 1),
                )
                for x in ch
            )
        )
        
        self.cv5 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc_action, 1),
                )
                for x in ch
            )
        )
        
        self.cv6 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc_target, 1),
                )
                for x in ch
            )
        )
    
    def bias_init(self):
        super().bias_init()
        self._stride_calculated = True
          
    def forward(self, x):
        if not hasattr(self, '_stride_calculated'):
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            return x

        instrument_scores = []
        action_scores = []
        target_scores = []
        
        for i in range(self.nl): # P3, P4, P5
            original_feat = x[i]
            
            cv2 = self.cv2[i](original_feat)
            cv3 = self.cv3[i](original_feat)
            cv4 = self.cv4[i](original_feat)
            cv5 = self.cv5[i](original_feat)
            cv6 = self.cv6[i](original_feat)

            instrument_scores.append(cv4.view(cv4.shape[0], self.nc_tool, -1))
            action_scores.append(cv5.view(cv5.shape[0], self.nc_action, -1))
            target_scores.append(cv6.view(cv6.shape[0], self.nc_target, -1))
            x[i] = torch.cat((cv2, cv3), 1)
               
        if self.training:  
            return x, instrument_scores, action_scores, target_scores
        
        y = self._inference(x)
        return y if self.export else y, x, instrument_scores, action_scores, target_scores


def register_mtl_detect():
    import ultralytics.nn.modules.head as head_module
    import ultralytics.nn.tasks as tasks_module
    import sys

    try:
        head_module.MTLDetect = MTLDetect

        tasks_module.MTLDetect = MTLDetect

        setattr(sys.modules['ultralytics.nn.tasks'], 'MTLDetect', MTLDetect)

        globals()['MTLDetect'] = MTLDetect

        tasks_module.__dict__['MTLDetect'] = MTLDetect

        if hasattr(head_module, '__all__'):
            if 'MTLDetect' not in head_module.__all__:
                if isinstance(head_module.__all__, tuple):
                    head_module.__all__ = list(head_module.__all__)
                head_module.__all__.append('MTLDetect')

        original_parse_model = tasks_module.parse_model
        
        def patched_parse_model(d, ch, verbose=True):
            import inspect
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_globals = frame.f_back.f_globals
                caller_globals['MTLDetect'] = MTLDetect
            
            return original_parse_model(d, ch, verbose)
        
        tasks_module.parse_model = patched_parse_model
        
        return True
        
    except Exception as e:
        print(f"Error: Could not register MTLDetect: {e}")
        import traceback
        traceback.print_exc()
        return False