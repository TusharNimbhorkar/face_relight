import scipy.io
import os
import numpy as np
from utils_normal import sh_cvt
cvt = sh_cvt()

dir1 = '/home/tushar/data/multipie_dlib_sfs/train_light/'
lights = os.listdir(dir1)

for light in lights:
    path= os.path.join(dir1,light)
    print(path)
    mat = scipy.io.loadmat(path)
    d = cvt.sfs2shtools(lighting=np.reshape(mat['light_out'],(3,9)))
    print(np.reshape(d[0],(9,1)))
    np.savetxt(path,np.reshape(d[0],(9,1)))

