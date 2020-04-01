'''
    construct shading using sh basis
'''
import numpy as np
from math import sin,cos
import matplotlib.pyplot as plt

def SH_basis(normal):
    '''
        get SH basis based on normal
        normal is a Nx3 matrix
        return a Nx9 matrix
        The order of SH here is:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
    '''
    normal = np.array(normal)
    numElem = normal.shape[0]

    norm_X = normal[:,0]
    norm_Y = normal[:,1]
    norm_Z = normal[:,2]

    sh_basis = np.zeros((numElem, 9))
    att= np.pi*np.array([1, 2.0/3.0, 1/4.0])
    sh_basis[:,0] = 0.5/np.sqrt(np.pi)*att[0]

    sh_basis[:,1] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Y*att[1]
    sh_basis[:,2] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Z*att[1]
    sh_basis[:,3] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_X*att[1]

    sh_basis[:,4] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_X*att[2]
    sh_basis[:,5] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_Z*att[2]
    sh_basis[:,6] = np.sqrt(5)/4/np.sqrt(np.pi)*(3*norm_Z**2-1)*att[2]
    sh_basis[:,7] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_X*norm_Z*att[2]
    sh_basis[:,8] = np.sqrt(15)/4/np.sqrt(np.pi)*(norm_X**2-norm_Y**2)*att[2]
    return sh_basis

def SH_basis_noAtt(normal):
    '''
        get SH basis based on normal
        normal is a Nx3 matrix
        return a Nx9 matrix
        The order of SH here is:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
    '''
    numElem = normal.shape[0]

    norm_X = normal[:,0]
    norm_Y = normal[:,1]
    norm_Z = normal[:,2]

    sh_basis = np.zeros((numElem, 9))
    sh_basis[:,0] = 0.5/np.sqrt(np.pi)

    sh_basis[:,1] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Y
    sh_basis[:,2] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Z
    sh_basis[:,3] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_X

    sh_basis[:,4] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_X
    sh_basis[:,5] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_Z
    sh_basis[:,6] = np.sqrt(5)/4/np.sqrt(np.pi)*(3*norm_Z**2-1)
    sh_basis[:,7] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_X*norm_Z
    sh_basis[:,8] = np.sqrt(15)/4/np.sqrt(np.pi)*(norm_X**2-norm_Y**2)
    return sh_basis

def get_shading(normal, SH):
    '''
        get shading based on normals and SH
        normal is Nx3 matrix
        SH: 9 x m vector
        return Nxm vector, where m is the number of returned images
    '''
    sh_basis = SH_basis(normal)
    shading = np.matmul(sh_basis, SH)
    #shading = np.matmul(np.reshape(sh_basis, (-1, 9)), SH)
    #shading = np.reshape(shading, normal.shape[0:2])
    return shading

def SH_basis_debug(normal):
    '''
        get SH basis based on normal
        normal is a Nx3 matrix
        return a Nx9 matrix
        The order of SH here is:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
    '''
    numElem = normal.shape[0]

    norm_X = normal[:,0]
    norm_Y = normal[:,1]
    norm_Z = normal[:,2]

    sh_basis = np.zeros((numElem, 9))
    att= np.pi*np.array([1, 2.0/3.0, 1/4.0])
    # att = [1,1,1]
    sh_basis[:,0] = 0.5/np.sqrt(np.pi)*att[0]

    sh_basis[:,1] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Y*att[1]
    sh_basis[:,2] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Z*att[1]
    sh_basis[:,3] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_X*att[1]

    sh_basis[:,4] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_X*att[2]
    sh_basis[:,5] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_Z*att[2]
    sh_basis[:,6] = np.sqrt(5)/4/np.sqrt(np.pi)*(3*norm_Z**2-1)*att[2]
    sh_basis[:,7] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_X*norm_Z*att[2]
    sh_basis[:,8] = np.sqrt(15)/4/np.sqrt(np.pi)*(norm_X**2-norm_Y**2)*att[2]
    return sh_basis

def get_shading_debug(normal, SH):
    '''
        get shading based on normals and SH
        normal is Nx3 matrix
        SH: 9 x m vector
        return Nxm vector, where m is the number of returned images
    '''
    sh_basis = SH_basis_debug(normal)
    shading = np.matmul(sh_basis, SH)
    #shading = sh_basis*SH[0]
    return shading


def euler_to_dir3(yaw, roll, pitch):
    x = -cos(yaw) * sin(roll) * sin(pitch) - sin(yaw) * cos(pitch)
    y = -sin(yaw) * sin(roll) * sin(pitch) + cos(yaw) * cos(pitch)
    z = cos(roll) * sin(pitch)
    return np.array([x, y, z])

def euler_to_dir(X, Z, Y):
    '''
    Converts Euler angles to a direction vector. The axis are switched to match Blender coordinate system
    :param X:
    :param Z:
    :param Y:
    :return:
    '''
    Sx = sin(X)
    Sy = sin(Y)
    Sz = sin(Z)
    Cx = cos(X)
    Cy = cos(Y)
    Cz = cos(Z)

    m = np.zeros((3,3))

    ##XYZ
    # m[0,0] = Cy * Cz
    # m[0,1] = -Cy * Sz
    # m[0,2] = Sy
    # m[1,0] = Cz * Sx * Sy + Cx * Sz
    # m[1,1] = Cx * Cz - Sx * Sy * Sz
    # m[1,2] = -Cy * Sx
    # m[2,0] = -Cx * Cz * Sy + Sx * Sz
    # m[2,1] = Cz * Sx + Cx * Sy * Sz
    # m[2,2] = Cx * Cy

    ##XZY
    # m[0,0] = Cy * Cz
    # m[0,1] = -Sz
    # m[0,2] = Cz * Sy
    # m[1,0] = Sx * Sy + Cx * Cy * Sz
    # m[1,1] = Cx * Cz
    # m[1,2] = -Cy * Sx + Cx * Sy * Sz
    # m[2,0] = -Cx * Sy + Cy * Sx * Sz
    # m[2,1] = Cz * Sx
    # m[2,2] = Cx * Cy + Sx * Sy * Sz

    ##YZX
    m[0,0] = Cy * Cz
    m[0,1] = Sx * Sy - Cx * Cy * Sz
    m[0,2] = Cx * Sy + Cy * Sx * Sz
    m[1,0] = Sz
    m[1,1] = Cx * Cz
    m[1,2] = -Cz * Sx
    m[2,0] = -Cz * Sy
    m[2,1] = Cy * Sx + Cx * Sy * Sz
    m[2,2] = Cx * Cy - Sx * Sy * Sz


    ##YXZ
    # m[0,0] = Cy * Cz + Sx * Sy * Sz
    # m[0,1] = Cz * Sx * Sy - Cy * Sz
    # m[0,2] = Cx * Sy
    # m[1,0] = Cx * Sz
    # m[1,1] = Cx * Cz
    # m[1,2] = -Sx
    # m[2,0] = -Cz * Sy + Cy * Sx * Sz
    # m[2,1] = Cy * Cz * Sx + Sy * Sz
    # m[2,2] = Cx * Cy

    dir = np.dot(m, [0,-1,0])
    dir  = dir/np.linalg.norm(dir)
    dir = dir[[0,2,1]]
    return dir


def gen_half_sphere(sh):
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x ** 2 + z ** 2)
    valid = mag <= 1
    y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    '''sh_viz'''
    # y = torch.Tensor.cpu(sh).detach().numpy()
    # sh = np.squeeze(y)
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
    shading = (shading * 255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid
    return shading

def show_half_sphere(sh):
    img = gen_half_sphere(sh)
    plt.imshow(img)
    plt.show()