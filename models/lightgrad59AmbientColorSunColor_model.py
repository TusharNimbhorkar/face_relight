# MSE-SUM on SH is mean not sum
# Default model
import torch
import sys
from .lightgrad59_model import lightgrad59Model

sys.path.append('.')

class lightgrad59AmbientColorSunColorModel(lightgrad59Model):
    def name(self):
        return 'lightgrad59AmbientColorSunColorModel'

    def _set_model_parameters(self):
        self.nc_light_extra = 6
