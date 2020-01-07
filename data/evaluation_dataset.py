import os.path
from data.base_dataset import BaseDataset, get_simple_transform
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch

def norm_img(img):
    resized_img = cv2.resize(img, (512, 512))
    lab_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)
    l_img = lab_img[:, :, 0]
    normed_img = l_img.astype(np.float32) / 255.0  # totensor also dividing????
    normed_img = normed_img.transpose((0, 1))
    normed_img = normed_img[..., None]
    return normed_img


def get_simple_transform(grayscale=False):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    # transform_list += [transforms.ToTensor(),
    #                    transforms.Normalize((0.5, 0.5, 0.5),
    #                                         (0.5, 0.5, 0.5))]
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)

class EvaluationDataset:
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--ffhq', type=int, default=70000, help='sample size ffhq')
        return parser

    def __init__(self, data_root_dir, pair_list_path):
        self.dict_AB = {}
        self.list_AB = []
        self.valid_id = []
        with open(pair_list_path) as f:
            lines = f.read().splitlines()

        for line in lines:
            self.list_AB.append([os.path.join(data_root_dir, line.split(' ')[0], line.split(' ')[1]),
                                 os.path.join(data_root_dir, line.split(' ')[0], line.split(' ')[2])])
            self.valid_id.append(line.split(' ')[0])

        self.valid_id = list(set(self.valid_id))

        self.pairs_it = iter(self.list_AB)
        print(len(self.list_AB))

        self.transform_A = get_simple_transform(grayscale=False)

    def __getitem__(self, item):

            sample = next(self.pairs_it)

            AB_path = sample
            A = cv2.imread(AB_path[0])
            inputA = norm_img(A)

            target_path = AB_path[1]
            B = cv2.imread(target_path)
            inputB = norm_img(B)

            # TODO NORMALISE??????? Check base dataset
            A = self.transform_A(inputA)
            B = self.transform_A(inputB)

            del_item = AB_path[0].split('_')[-1][:-4]
            target_item = target_path.split('_')[-1][:-4]

            # TODO: LIGHT!!!!!
            AL_path = AB_path[0][:-6] + 'light_' + del_item + '.txt'
            sh = np.loadtxt(AL_path)
            sh = sh[0:9]
            sh_AL = sh * 1.0
            sh_AL = np.squeeze(sh_AL)
            sh_AL = np.reshape(sh_AL, (9, 1, 1)).astype(np.float32)

            BL_path = AB_path[0][:-6] + 'light_' + target_item + '.txt'
            sh = np.loadtxt(BL_path)
            sh = sh[0:9]
            sh_BL = sh * 1.0  # 0.7
            sh_BL = np.squeeze(sh_BL)
            sh_BL = np.reshape(sh_BL, (9, 1, 1)).astype(np.float32)
            return {'A': A, 'B': B, 'C': np.NaN,'D':np.NaN, 'AL': torch.from_numpy(sh_AL), 'BL': torch.from_numpy(sh_BL),
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.list_AB)

    def name(self):
        return 'EvaluationDataset'


class EvaluationDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def name(self):
        return 'CustomDatasetDataLoader'

    def __init__(self, data_root_dir, pair_list_path, n_batch, n_threads):
        self.n_batch = n_batch
        self.dataset = EvaluationDataset(data_root_dir, pair_list_path)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.n_batch,
            shuffle=False,
            num_workers=int(n_threads),
            # pin_memory=True
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.n_batch >= len(self.dataset):
                print("DEBUG", len(data))
                break
            yield data
