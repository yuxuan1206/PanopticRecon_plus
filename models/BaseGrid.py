'''
Author: yuxuan1206 610939662@qq.com
Date: 2022-06-02 21:44:26
LastEditors: yuxuan1206 610939662@qq.com
LastEditTime: 2022-08-17 18:32:06
FilePath: /occuSLAM3D_indoor/models/BaseGrid.py
Description: 

Copyright (c) 2022 by yuxuan1206 610939662@qq.com, All Rights Reserved. 
'''
import torch
import torch.nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from .rays import *
import os

class BaseGrid:
    def __init__(self, device, config):
        super(BaseGrid, self).__init__()
        self.device = device

        self.N_sample = config['params']['N_sample']
        self.near, self.far = config['params']['near'], config['params']['far']
        self.grid_resolution = config['params']['grid_resolution']
        self.important_sample = 1
        self.is_optim_depth = False
        self.is_optim_rgb = False
        
    def set_optim_flag(self, is_optim_depth, is_optim_rgb):
        self.is_optim_depth = is_optim_depth
        self.is_optim_rgb = is_optim_rgb
        
    def set_important_sample(self, flag):
        self.important_sample = flag
        
    def load_trans(self, path):
        self.trans_world_to_scale = torch.FloatTensor(np.loadtxt(path)).to(self.device) 
        