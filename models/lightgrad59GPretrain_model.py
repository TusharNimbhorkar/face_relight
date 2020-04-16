# MSE-SUM on SH is mean not sum
# Default model
import torch
import sys
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

sys.path.append('.')
# from relight_model import *
from .skeleton512 import *
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from models.lightgrad59_model import lightgrad59Model
#


class lightgrad59GPretrainModel(lightgrad59Model):
    def name(self):
        return 'lightgrad59GPretrainModel'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs_G_only=2