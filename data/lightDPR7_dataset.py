# Our dataloader.
import json
import os.path
import random
from json import JSONDecodeError

from data.base_dataset import BaseDataset, get_simple_transform
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from commons.common_tools import Logger, BColors
from copy import deepcopy

from models.common.data import img_to_input

log = Logger("DataLoader", tag_color=BColors.LightBlue)

class lightDPR7Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.add_argument('--enable_neutral', action='store_true', help='Enable or disable input target sh')
        parser.add_argument('--n_ids', type=int, default=None, help='Select the amount of identities to take from the dataset. If not used, taking all identities.')
        parser.add_argument('--input_mode', type=str, default='L', choices=['L', 'LAB', 'RGB'], help='Choose between L, LAB and RGB input')
        parser.add_argument('--exclude_props', type=str, default='',
                            help='Define keys which to exclude when passing to the light network')
        parser.add_argument('--enable_altered_orig', action='store_true',
                            help='Enable per relit image corresponding original image. If not enabled the target will always be the initial original image. Used to preserve some variability in the target - like ambient color.')
        parser.add_argument('--enable_stack', action='store_true', help='Enable or disable stack input sh')
        if is_train:
            parser.add_argument('--ffhq', type=int, default=70000, help='sample size ffhq')
            parser.add_argument('--force_ambient_intensity', action='store_true', help='Do not use ambient color')

        return parser

    def _random_derangement(self, n):
        while True:
            v = list(range(n))
            for j in range(n - 1, -1, -1):
                p = random.randint(0, j)
                if v[p] == j:
                    break
                else:
                    v[j], v[p] = v[p], v[j]
            else:
                if v[0] != 0:
                    return tuple(v)

    def _get_all_derangements(self, n):

        # enumerate all derangements for testing
        import itertools
        counter = {}
        for p in itertools.permutations(range(n)):
            if all(p[i] != i for i in p):
                counter[p] = 0

        # make M probes for each derangement
        M = 5000
        for _ in range(M * len(counter)):
            # generate a random derangement
            p = self._random_derangement(n)
            # is it really?
            assert p in counter
            # ok, record it
            counter[p] += 1

        # the distribution looks uniform
        return [p for p, _ in sorted(counter.items())]

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.enable_target = not opt.enable_neutral
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths_ = make_dataset(self.dir_AB, n_ids=opt.n_ids)
        self.dict_AB = {}
        self.list_AB = []

        self.excluded_props = self.opt.exclude_props.split(" ")
        if len(self.excluded_props) == 1 and self.excluded_props[0]=='':
            self.excluded_props=[]

        self.n_real = len(os.listdir(os.path.join(self.opt.dataroot,'real_im')))
        self.img_size = self.opt.img_size
        self.use_segments = self.opt.segment
        self.input_mode = self.opt.input_mode
        self.force_ambient_intensity = self.opt.force_ambient_intensity
        synth_n_per_id = self.opt.n_synth
        n_want = self.opt.n_first

        # der = self._get_all_derangements(n_per_id)

        for i in range(0, len(self.AB_paths_)):

            orig_img_path = self.AB_paths_[i]['orig']
            sample_paths = self.AB_paths_[i]['synth'][:n_want]
            sample_paths.append(orig_img_path)

            orig_altered = []
            if opt.enable_altered_orig:
                orig_altered = self.AB_paths_[i]['orig_altered']

            pairs = self.gen_sample_pairs(sample_paths, n_want+1, altered_orig_paths=orig_altered)

            self.list_AB.extend(pairs)
            # print(pairs)
            # print()
        random.shuffle(self.list_AB)

        assert(opt.resize_or_crop == 'resize_and_crop')  # only support this mode
        assert(self.opt.loadSize >= self.opt.fineSize)

        self.transform_A = get_simple_transform(grayscale=False)

    def gen_sample_pairs(self, sample_paths, n_per_id, altered_orig_paths=None):
        # i2 = deepcopy(sample_paths)
        orig_img_path = sample_paths[-1]
        src_paths = sample_paths

        if altered_orig_paths is None:
            altered_orig_paths = []

        pairs = []
        if self.enable_target:
            np.random.shuffle(src_paths)
            pairs.append([orig_img_path, orig_img_path])
            sel_der = self._random_derangement(n_per_id-1)
            tgt_paths = [src_paths[i] for i in sel_der]
        else:
            if len(altered_orig_paths) > 0:
                p = np.random.permutation(len(altered_orig_paths))
                src_paths = [src_paths[i] for i in p]
                tgt_paths = [altered_orig_paths[i] for i in p]

                # tgt_paths.append(orig_img_path)
                # src_paths.append(altered_orig_paths[np.random.randint(len(altered_orig_paths))])
                # i2 = [orig_img_path] * len(src_paths)
            else:
                np.random.shuffle(src_paths)
                tgt_paths = [orig_img_path] * len(src_paths)

        for src_path, tgt_path in zip(src_paths, tgt_paths):

            # blist = list(set(i2) - set([src_path]))
            #
            # tgt_path = blist.pop(np.random.randint(len(blist)))

            # if src_path in i2:
            #     blist.append(src_path)
            # else:
            #     print('yep')
            #
            # i2 = blist

            # print([p.rsplit('/',1)[-1] for p in blist])

            # print(src_path, tgt_path)
            pairs.append([src_path, tgt_path])

        # print()

        return pairs

    def _get_paths(self, real_img_id, input_img_id):
        AB_path = self.list_AB[input_img_id]
        source_path = AB_path[0]
        target_path = AB_path[1]

        src_light_path = AB_path[0][:-6] + 'light_%s.txt' % AB_path[0].split('_')[-1][:-4]
        tgt_light_path = AB_path[0][:-6] + 'light_%s.txt' % target_path.split('_')[-1][:-4]

        real_img_path = os.path.join(self.opt.dataroot,'real_im',"{:05d}".format(real_img_id)+'.png')
        orig_img_path = target_path[:-5]+'5.png'
        segment_img_path = os.path.join(self.opt.dataroot, 'segments', AB_path[0].split('/')[-1].split('_')[0],
                     source_path.split('/')[-1].split('_')[0] + '.png')

        return real_img_path, orig_img_path, source_path, target_path, segment_img_path, src_light_path, tgt_light_path

    def _img_to_input(self, img, input_mode=None):

        if input_mode is None:
            input_mode = self.input_mode

        input = img_to_input(img, input_mode, transform=self.transform_A)

        return input

    def _get_sh_legacy(self, sh):
        # sh = sh[0:9]
        sh = sh * 1.0
        sh = np.squeeze(sh)
        sh = np.reshape(sh, (len(sh), 1, 1)).astype(np.float32)
        return sh

    def _read_img(self, img_path, size):
        img = cv2.imread(img_path)
        try:
            if size != img.shape[0] or size != img.shape[1]:
                img = cv2.resize(img, (size, size))
        except Exception as e:
            print(img_path)
            raise e

        return img

    def _get_light_data(self, props):
        '''
        Generates a numpy array to be used when working with the Light network
        :param props: Properties to convert to numpy array
        :return:
        '''

        # enable_sun_color = 'sun_color' in props.keys()
        # enable_ambient_color = 'ambient_color' in props.keys()
        # enable_ambient_intensity = 'ambient_intensity' in props.keys()
        # enable_sun_intensity = 'sun_intensity' in props.keys()
        # enable_sun_diameter = 'sun_diameter' in props.keys()
        #


        light_data = props['sh']
        keys = list(props.keys())
        if 'sun_intensity' in keys:
            light_data[0] = props['sun_intensity']
            keys.remove("sun_intensity")

        keys.remove("sh")

        if self.force_ambient_intensity:
            key_order = ['ambient_intensity', 'ambient_color', 'sun_diameter', 'sun_color']
        else:
            key_order = ['ambient_intensity', 'sun_diameter', 'sun_color', 'ambient_color']

        keys_ordered = [key for key in key_order if key in keys]
        for key in keys_ordered:
            keys.remove(key)

        keys = keys_ordered + keys

        for key in self.excluded_props:
            if not self.force_ambient_intensity:
                if key not in keys:
                    raise ValueError("Key %s not found" % key)

            if key in keys:
                keys.remove(key)

        for key in keys:
            if self.force_ambient_intensity and key == 'ambient_color':
                light_data.append(props[key][0])
            elif isinstance(props[key], list):
                light_data.extend(props[key])
            else:
                light_data.append(props[key])

        light_data = np.array(light_data).reshape((-1,1,1))
        # log.d('TEST', keys)
        return light_data

    def __read_properties(self, path):

        try:
            with open(path) as light_file:
                props = json.load(light_file)
                props = self._get_light_data(props)
        except JSONDecodeError:
            sh = np.loadtxt(path)
            props = self._get_sh_legacy(sh)

        return props


    def __getitem__(self, index):
        real_img_id = random.choice(range(0, self.n_real))
        AB_path = self.list_AB[index]
        real_path, orig_path, source_path, target_path, segment_img_path, src_light_path, tgt_light_path = self._get_paths(real_img_id, index)

        #Real discriminator image
        img_real = cv2.imread(real_path)

        try:
            if self.img_size < img_real.shape[0]:
                img_real = cv2.resize(img_real, (self.img_size, self.img_size))
        except AttributeError as e:
            print(real_path)
            raise e
        except Exception as e:
            print(real_path)
            raise e

        input_real = self._img_to_input(img_real)

        #Real original image
        img_orig = self._read_img(orig_path, self.img_size)
        input_orig = self._img_to_input(img_orig)

        #Segments
        back_original = None
        if self.use_segments:
            segment_im = cv2.imread(segment_img_path)
            segment_im=cv2.resize(segment_im,(self.img_size,self.img_size))

            segment_im[segment_im==255]=1
            segment_im_invert = np.invert(segment_im)
            segment_im_invert[segment_im_invert == 254] = 0
            segment_im_invert[segment_im_invert == 255] = 1

            back_original = np.multiply(img_orig, segment_im_invert)

        #Source image
        img_src = self._read_img(source_path, self.img_size)

        if self.use_segments:
            img_src = back_original + np.multiply(img_src, segment_im)

        input_src = self._img_to_input(img_src)

        #Target image
        img_tgt = self._read_img(target_path, self.img_size)

        if self.use_segments:
            img_tgt = back_original + np.multiply(img_tgt, segment_im)

        input_tgt = self._img_to_input(img_tgt)

        #Source light

        props_src = self.__read_properties(src_light_path).astype(np.float32)
        props_tgt = self.__read_properties(tgt_light_path).astype(np.float32)

        data_dict = {'A': input_src, 'B': input_tgt,'C':input_real,'D':input_orig,
                     'AL':torch.from_numpy(props_src),'BL':torch.from_numpy(props_tgt),
                     'A_paths': AB_path, 'B_paths': AB_path}


        if self.opt.enable_stack:
            orig_path_sh = orig_path.replace('orig.png', 'light_orig_sh.txt')
            orig_sh = self.__read_properties(orig_path_sh).astype(np.float32)
            data_dict['orig_sh'] = torch.from_numpy(orig_sh)


        return data_dict
    def __len__(self):
        return len(self.list_AB)

    def name(self):
        return 'lightDPR7Dataset'
