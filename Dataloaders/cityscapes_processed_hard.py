import glob
import json
import multiprocessing.dummy as multiprocessing
import os.path as osp
import random
import numpy as np
import skimage.transform as transform
import torch
from torch.utils.data import Dataset
import Utils.utils as utils

EPS = 1e-7

def process_info(args):
    """
    Process a single json file
    """
    fname, opts = args

    with open(fname, 'r') as f:
        ann = json.load(f)

    examples = []
    skipped_instances = 0

    for instance in ann:
        components = instance['components']

        if opts['class_filter'] is not None and instance['label'] not in opts['class_filter']:
            continue

        candidates = [c for c in components if len(c['poly']) >= opts['min_poly_len']]

        if opts['sub_th'] is not None:
            total_area = np.sum([c['area'] for c in candidates])
            candidates = [c for c in candidates if c['area'] > opts['sub_th'] * total_area]

        candidates = [c for c in candidates if c['area'] >= opts['min_area']]

        if opts['skip_multicomponent'] and len(candidates) > 1:
            skipped_instances += 1
            continue

        instance['components'] = candidates
        if candidates:
            examples.append(instance)

    return examples, skipped_instances


def collate_fn(batch_list):
    keys = batch_list[0].keys()
    collated = {}

    for key in keys:
        val = [item[key] for item in batch_list]

        t = type(batch_list[0][key])

        if t is np.ndarray:
            try:
                val = torch.from_numpy(np.stack(val, axis=0))
            except:
                # for items that are not the same shape
                # for eg: orig_poly
                val = [item[key] for item in batch_list]

        collated[key] = val

    return collated

class DataProvider(Dataset):
    """
    Class for the data provider
    """

    def __init__(self, opts, split='train', mode='train_ce', debug=False):
        """
        split: 'train', 'train_val' or 'val'
        opts: options from the json file for the dataset
        """
        self.opts = opts
        self.mode = mode
        self.debug = debug
        print self.opts.keys()
        print 'Dataset Options: ', opts

        if self.mode != 'tool':
            # in tool mode, we just use these functions
            self.data_dir = osp.join(opts['data_dir'], split)
            self.instances = []
            self.read_dataset()
            print 'Read %d instances in %s split' % (len(self.instances), split)

    def read_dataset(self):
        data_list = glob.glob(osp.join(self.data_dir, '*/*.json'))
        data_list = [[d, self.opts] for d in data_list]
        if self.debug:
            data_list = data_list[:20]
        pool = multiprocessing.Pool(self.opts['num_workers'])
        data = pool.map(process_info, data_list)
        pool.close()
        pool.join()

        print "Dropped %d multi-component instances" % (np.sum([s for _, s in data]))

        self.instances = [instance for image, _ in data for instance in image]

        if self.debug:
            self.instances = self.instances[:2]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.prepare_instance(idx)

    def prepare_instance(self, idx):
        """
        Prepare a single instance, can be both multicomponent
        or just a single component
        """
        instance = self.instances[idx]

        if self.opts['skip_multicomponent']:
            # Skip_multicomponent is true even during test because we use only
            # 1 bbox and no polys
            assert len(instance['components']) == 1, 'Found multicomponent instance\
            with skip_multicomponent set to True!'

            component = instance['components'][0]
            results = self.prepare_component(instance, component)

            if 'test' in self.mode:
                results['instance'] = instance

        else:
            if 'test' in self.mode:
                component = instance['components'][0]
                results = self.prepare_component(instance, component)
            elif 'train' in self.mode:
                component = random.choice(instance['components'])
                results = self.prepare_component(instance, component)

            results['instance'] = instance
            # When we have multicomponents turned on, also send the whole instance
            # In test, this is used to calculate IoU. In train(RL/Evaluator),
            # this is used to calculate the reward

        return results

    def prepare_component(self, instance, component):
        """
        Prepare a single component within an instance
        """
        lo, hi = self.opts['random_context']
        context_expansion = random.uniform(lo, hi)

        crop_info = self.extract_crop(component, instance, context_expansion)

        img = crop_info['img']

        train_dict = {}

        # for Torch, use CHW, instead of HWC
        img = img.transpose(2, 0, 1)
        # blank_image
        return_dict = {
            'img': img,
            'img_path': instance['img_path'],
            'context_expansion': context_expansion
        }

        return_dict.update(train_dict)

        return return_dict

    def extract_crop(self, component, instance, context_expansion):
        img = utils.rgb_img_read(instance['img_path'])

        bbox = instance['bbox']
        x0, y0, w, h = bbox

        x_center = x0 + (1 + w) / 2.
        y_center = y0 + (1 + h) / 2.

        widescreen = True if w > h else False

        if not widescreen:
            img = img.transpose((1, 0, 2))
            x_center, y_center, w, h = y_center, x_center, h, w

        x_min = int(np.floor(x_center - w * (1 + context_expansion) / 2.))
        x_max = int(np.ceil(x_center + w * (1 + context_expansion) / 2.))

        x_min = max(0, x_min)
        x_max = min(img.shape[1] - 1, x_max)

        patch_w = x_max - x_min
        # NOTE: Different from before

        y_min = int(np.floor(y_center - patch_w / 2.))
        y_max = y_min + patch_w

        top_margin = max(0, y_min) - y_min

        y_min = max(0, y_min)
        y_max = min(img.shape[0] - 1, y_max)

        scale_factor = float(self.opts['img_side']) / patch_w

        patch_img = img[y_min:y_max, x_min:x_max, :]

        new_img = np.zeros([patch_w, patch_w, 3], dtype=np.float32)
        new_img[top_margin: top_margin + patch_img.shape[0], :, ] = patch_img

        new_img = transform.rescale(new_img, scale_factor, order=1,
                                    preserve_range=True, multichannel=True)
        new_img = new_img.astype(np.float32)
        # assert new_img.shape == [self.opts['img_side'], self.opts['img_side'], 3]

        starting_point = [x_min, y_min - top_margin]

        if not widescreen:
            # Now that everything is in a square
            # bring things back to original mode
            new_img = new_img.transpose((1, 0, 2))
            starting_point = [y_min - top_margin, x_min]

        return_dict = {
            'img': new_img,
            'patch_w': patch_w,
            'top_margin': top_margin,
            'patch_shape': patch_img.shape,
            'scale_factor': scale_factor,
            'starting_point': starting_point,
            'widescreen': widescreen
        }
        return return_dict
