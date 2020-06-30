#!/usr/bin/python3
"""
https://stackoverflow.com/questions/550470/overload-print-python

In this function we will filter out blender output by piping the stdout, and filtering based ont he TAG
"""
from __future__ import print_function

try:
    import __builtin__
except ImportError:
    # Python 3
    import builtins as __builtin__
    
    
TAG = "#"
TAG_bytes = TAG.encode('utf-8')
def print(*args, **kwargs):
    return __builtin__.print(TAG, *args, **kwargs, flush=True)
