import time
import torch

class Chronometer:
    def __init__(self, device):
        self.device = device
        self.device_str = str(device)
        if self.device_str == 'cpu':
            self.start_time = time.time()
            self.end_time = time.time()
        else:
            self.start_time = torch.cuda.Event(enable_timing=True)
            self.end_time = torch.cuda.Event(enable_timing=True)

    def tick(self):
        '''
        Resets the chronometer
        :return:
        '''
        if self.device_str== 'cpu':
            self.start_time = time.time()
        else:
            self.start_time.record()

    def tock(self):
        '''
        Returns elapsed time in seconds
        :return:
        '''
        if self.device_str== 'cpu':
            self.end_time = time.time()
            elapsed = self.end_time - self.start_time
        else:
            self.end_time.record()
            torch.cuda.synchronize(self.device)
            elapsed = self.start_time.elapsed_time(self.end_time)

        return elapsed/1000