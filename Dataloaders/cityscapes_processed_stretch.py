import json
import multiprocessing.dummy as multiprocessing
import PIL
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import cv2

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

def collate_fn(batch_list):
    keys = batch_list[0].keys()
    collated = {}
    all_valid = [np.array(b['valid']) for b in batch_list]
    all_valid = np.concatenate(all_valid)
    collated['valid'] = all_valid

    for key in keys:
        if key == 'valid': continue
        val = [item[key] for item in batch_list]
        t = type(batch_list[0][key])

        if t is np.ndarray:
            try:
                val = torch.from_numpy(np.stack(val, axis=0))
            except:
                val = [item[key] for item in batch_list]
        if t is torch.Tensor:
            try:
                val = torch.cat([v for v in val], dim = 0)
            except:
                val = [item[key] for item in batch_list]

        collated[key] = val

    return collated

def process_info(fname, skip_multicomp=False):
    with open(fname, 'r') as f:
        ann = json.load(f)

    ret = []
    idx = 0
    for obj in ann:
        if obj['label'] not in [
            "car",
            "truck",
            "train",
            "bus",
            "motorcycle",
            "bicycle",
            "rider",
            "person"
        ]:
            continue

        components = obj['components']
        candidates = [c for c in components if len(c['poly']) >= 3]
        candidates = [c for c in candidates if c['area'] >= 100]

        instance = dict()
        instance['area'] = [c['area'] for c in candidates]
        instance['gt_polygon'] = [np.array(comp['poly']) for comp in candidates]
        instance['im_size'] = (obj['img_height'], obj['img_width'])
        instance['im_path'] = obj['img_path']
        instance['label'] = obj['label']
        instance['idx'] = str(idx)
        idx += 1

        if skip_multicomp and len(candidates) > 1:
            continue
        if candidates:
            ret.append(instance)

    return ret

class CityScapesProcessedStretchMulticomp(Dataset):
    """CityScapes dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True, split='train',
                 db_root_dir=None,
                 transform=None,
                 min_poly=3,
                 random_select=False):

        self.train = train
        self.split = split
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.min_poly = min_poly
        self.ann_list = self.get_ann_list(min_poly)
        self.random_select = random_select

    def __len__(self):
        return len(self.ann_list)

    def __getitem__(self, idx):
        ann = self.ann_list[idx]
        img = np.array(PIL.Image.open(ann['im_path']).convert('RGB')).astype(np.float32)
        gt_list = []

        for idx, poly in enumerate(ann['gt_polygon']):
            gt = np.zeros(ann['im_size'])
            gt = cv2.fillPoly(gt, [poly], 1)
            gt_list.append(gt)

        if self.random_select:
            n_all = len(ann['gt_polygon'])
            select_idx = np.random.choice(n_all)
            gt_polygon = [ann['gt_polygon'][select_idx]]
            gt_list = [gt_list[select_idx]]
        else:
            gt_polygon = ann['gt_polygon']

        sample = {'image': img, 'gt': gt_list, 'gt_polygon': gt_polygon}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_ann_list(self, min_poly=3):
        root = ''
        if not os.path.exists(root):
            os.makedirs(root)
        ann_list_path = os.path.join(root, self.split + '.npy')

        if os.path.exists(ann_list_path):
            return np.load(ann_list_path).tolist()
        else:
            print("Preprocessing of CityScapes Dataset. This would be done only once. ")
        data_dir = os.path.join(self.db_root_dir, self.split)
        ann_path_list = recursive_glob(data_dir, suffix='.json')

        pool = multiprocessing.Pool(32)
        ann_list = pool.map(process_info, ann_path_list)
        ann_list = [obj for ann in ann_list for obj in ann]
        print('==> Saving')
        np.save(ann_list_path, ann_list)
        return ann_list
