import os.path
from .lightDPR7_dataset import lightDPR7Dataset

class light3DULABDataset(lightDPR7Dataset):
    '''
    A dataloader to work with the 3DU face relight dataset
    '''

    def __init__(self, opt):
        super(light3DULABDataset,self).__init__(opt)
        self.use_large_real_set = os.path.exists(os.path.join(self.opt.dataroot, 'real_im', "{:06d}".format(0) + '.png'))
        if self.opt.load_realD2 is not None:
            self.use_large_real_set2 = os.path.exists(os.path.join(self.opt.load_realD2, "{:06d}".format(0) + '.png'))

    def _get_item_path(self, path, fname_pattern):
        dir, fname = path.rsplit('/',1)
        id = fname.split('_')[-1][:-4]
        path = os.path.join(dir, fname_pattern % id)
        return path

    def _get_paths(self, real_img_id, input_img_id, real_img_id2):
        AB_path = self.list_AB[input_img_id]
        source_path = AB_path[0]
        target_path = AB_path[1]

        src_light_path = self._get_item_path(source_path, 'light_%s_sh.txt')
        tgt_light_path = self._get_item_path(target_path, 'light_%s_sh.txt')

        if self.use_large_real_set:
            real_img_path = os.path.join(self.opt.dataroot,'real_im',"{:06d}".format(real_img_id)+'.png')
        else:
            real_img_path = os.path.join(self.opt.dataroot, 'real_im', "{:05d}".format(real_img_id) + '.png')

        if real_img_id2 is not None:
            if self.use_large_real_set2:
                real_img_path2 = os.path.join(self.opt.load_realD2, "{:06d}".format(real_img_id2) + '.png')
            else:
                real_img_path2 = os.path.join(self.opt.load_realD2, "{:05d}".format(real_img_id2) + '.png')

        orig_img_path = os.path.join(target_path.rsplit('/',1)[0], 'orig.png')
        segment_img_path = os.path.join(self.opt.dataroot,'segments',source_path.split('/')[-2],AB_path[0].split('/')[-2]+'.png')
        if real_img_id2 is not None:
            return real_img_path, orig_img_path, source_path, target_path, segment_img_path, src_light_path, tgt_light_path, real_img_path2
        else:
            return real_img_path, orig_img_path, source_path, target_path, segment_img_path, src_light_path, tgt_light_path


    def name(self):
        return 'light3DULABDataset'
