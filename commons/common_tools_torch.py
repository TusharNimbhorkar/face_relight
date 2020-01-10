# ==========================================
#  Project:  Common Tools
#  Author: Nedko Savov
#  Date: 9 January 2020
# ==========================================

import torch
import time
from .common_tools import Logger, BColors

log = Logger("Torch Commons", tag_color=BColors.LightYellow)

class Chronometer:
    '''
    A simple class to measure time in seconds for computation both for cpu and gpu
    '''

    def __init__(self, device='cpu'):
        '''
        :param device: "cpu", "gpu" or "cuda" (last two are equivalent) The strings can be followed by :<extra_input> which will be ignored (e.g. "cuda:0")
        '''

        device_str = str(device).split(':')[0]
        if device_str == 'cuda':
            device_str = 'gpu'

        assert(device_str in ['cpu', 'gpu'])

        self.device = torch.device(str(device))
        self.device_str = device_str
        self.active=False
        self.time_start = 0

        if self.device_str == 'gpu':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)


    def start(self):
        self.active=True

        if self.device_str == 'cpu':
            self.time_start = time.time()
        elif self.device_str == 'gpu':
            self.start_event.record()

    def stop(self):

        if self.device_str == 'cpu':
            self.time_end = time.time()
            return (self.time_end-self.time_start)
        elif self.device_str == 'gpu':
            self.end_event.record()
            torch.cuda.synchronize(self.device)
            return self.start_event.elapsed_time(self.end_event)/1000

