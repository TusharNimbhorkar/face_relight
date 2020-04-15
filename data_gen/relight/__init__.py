#!/usr/bin/python3
import os
import glob
from .bprint import TAG, TAG_bytes
from os.path import join, basename
from subprocess import Popen, PIPE, STDOUT

blender_relight_py = join(os.path.dirname(os.path.abspath(__file__)), "relight.py")

def dump(obj):
   for attr in dir(obj):
       if hasattr( obj, attr ):
           print( "obj.%s = %s" % (attr, getattr(obj, attr)))

def relight(image_path, normal_path, dest_folder, blender_name="datagen.blend", print_command=True):
    blender_relight_blend = join(os.path.dirname(os.path.abspath(__file__)), blender_name)
    command = "blender {} -b -P {} -- {} {} {}".format(blender_relight_blend, blender_relight_py, image_path, normal_path, dest_folder)
    print(command)
    if print_command:
        print(command)
    p = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        if not line:
            break
        if line.startswith(TAG_bytes):
            if line.endswith(b"__r\n"):
                print(line[(len(TAG_bytes)+1):-4].decode('utf-8').rstrip(), end='\r')
            else:
                print(line[(len(TAG_bytes)+1):].decode('utf-8').rstrip())
    p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"blender exited with {p.returncode}")
