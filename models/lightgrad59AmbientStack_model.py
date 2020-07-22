# MSE-SUM on SH is mean not sum
# Default model
import torch
import sys
from .lightgrad59stack_model import lightgrad59stackModel as lightgrad59Model

sys.path.append('.')

class lightgrad59AmbientStackModel(lightgrad59Model):
    def name(self):
        return 'lightgrad59AmbientStackModel'

    def _set_model_parameters(self):
        self.nc_light_extra = 1
