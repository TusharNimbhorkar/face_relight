"""Modified Image folder class
Code from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
Modified the original code so that it also loads images from the current
directory as well as the subdirectories
"""

import torch.utils.data as data

from commons.common_tools import sort_numerically
from PIL import Image
import os
import os.path
import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, n_ids=None, n_per_dir=-1):
    img_paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    entry_dirs = glob.glob(os.path.join(dir,'*'))
    sort_numerically(entry_dirs)
    entry_dirs = entry_dirs[:n_ids]


    for entry_dir in entry_dirs:
        paths_altered_orig = []
        path_orig = None
        paths = sorted(glob.glob(os.path.join(entry_dir,'*')))
        # paths = [path for path in paths if is_image_file(path)]


        paths_synth = []
        for path in paths:
            if is_image_file(path):
                fname = path.rsplit('/',1)[-1].rsplit('.',1)[0]
                if fname == 'orig':
                    path_orig = path
                elif fname.rsplit('_', 1)[-1] == 'orig':
                    paths_altered_orig.append(path)
                else:
                    paths_synth.append(path)

        if n_per_dir > 0:
            paths_synth = paths_synth[:n_per_dir]
            paths_altered_orig = paths_altered_orig[:n_per_dir]

        assert path_orig is not None, 'No original image found in %s' % entry_dir
        assert len(paths_synth)>0, 'No synthetic images in %s' % entry_dir

        paths_dict = {'orig':path_orig,
                 'orig_altered':paths_altered_orig,
                 'synth':paths_synth,}

        img_paths.append(paths_dict)
    return img_paths


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
