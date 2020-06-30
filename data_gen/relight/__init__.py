#!/usr/bin/python3
import os
import glob
import time
from .bprint import TAG, TAG_bytes
from os.path import join, basename
from subprocess import Popen, PIPE, STDOUT

def dump(obj):
   for attr in dir(obj):
       if hasattr( obj, attr ):
           print( "obj.%s = %s" % (attr, getattr(obj, attr)))

def relight(data_root, dst_root, image_name, normals_name, size=256, world_color=0.17, intensity=4.2, sun_angle=0.785398, blender_name="datagen.blend", relight_name="relight.py", print_command=True):
    blender_relight_blend = join(os.path.dirname(os.path.abspath(__file__)), blender_name)
    blender_relight_py = join(os.path.dirname(os.path.abspath(__file__)), relight_name)
    restart = True
    start_idx = 0
    
    while restart:
        restart = False
        command = "blender {} -b -P {} -- {} {} {} {} {} {} {} {} {}".format(blender_relight_blend, blender_relight_py, data_root, dst_root, image_name, normals_name, start_idx, size, world_color, intensity, sun_angle)
        if print_command:
            print(command)
        
        t0 = time.time()
        p = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)
        for line in iter(p.stdout.readline, b''):
            if not line:
                break
            if line.startswith(TAG_bytes):
                if line.endswith(b"__break\n"):
                    print("Restarting....")
                    start_idx = int(line[(len(TAG_bytes)+1):-8].decode('utf-8').rstrip())
                    restart = True
                elif line.endswith(b"__r\n"):
                    print(line[(len(TAG_bytes)+1):-4].decode('utf-8').rstrip(), end='\r')
                else:
                    print(line[(len(TAG_bytes)+1):].decode('utf-8').rstrip())
        p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"blender exited with {p.returncode}")        
        
        t1 = time.time()
        print(f"time elapsed = {t1 - t0}s")
