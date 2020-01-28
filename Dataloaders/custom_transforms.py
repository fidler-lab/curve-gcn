import torch, cv2
import numpy as np
import Dataloaders.helpers as helpers

class CropFromMaskStretchMulticomp(object):
    def __init__(self, crop_elems=('image', 'gt', 'gt_polygon'),
                 mask_elem='gt',
                 relax=0,
                 zero_pad=False,
                 dummy=False):

        self.crop_elems = crop_elems
        self.mask_elem = mask_elem
        self.relax = relax
        self.zero_pad = zero_pad
        self.dummy = dummy

    def __call__(self, sample):
        _target_list = sample[self.mask_elem]
        sample['crop_image'] = []
        sample['crop_gt'] = []
        sample['crop_polygon'] = []
        sample['valid'] = []

        for i_comp, _target in enumerate(_target_list):
            _target = np.expand_dims(_target, axis=-1)
            elem = 'image'
            _img = sample[elem]
            if _img.ndim == 2:
                _img = np.expand_dims(_img, axis=-1)
            _crop_img = []
            for k in range(0, _target.shape[-1]):
                if np.max(_target[..., k]) == 0:
                    _crop_img.append(np.zeros(_img.shape, dtype=_img.dtype))
                else:
                    _tmp_target = _target[..., k]
                    _crop_img.append(helpers.crop_from_mask(_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
            sample['crop_image'].append(_crop_img[0])

            elem = 'gt'
            _img = sample[elem][i_comp]
            _crop_img = []
            if _img.ndim == 2:
                _img = np.expand_dims(_img, axis=-1)
            for k in range(0, _target.shape[-1]):
                _tmp_img = _img[..., k]
                _tmp_target = _target[..., k]
                if np.max(_target[..., k]) == 0:
                    _crop_img.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
                else:
                    _crop_img.append(helpers.crop_from_mask(_tmp_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
            sample['crop_gt'].append(_crop_img[0])

            elem = 'gt_polygon'
            gt_polygon = sample[elem][i_comp]
            valid = 1
            try:
                _crop_polygon = helpers.crop_from_mask_polygon(gt_polygon, _img, _target[..., 0], relax=self.relax, zero_pad=self.zero_pad)
            except:
                _crop_polygon = np.asarray([[0.5,0.5],[0.5,0.6],[0.6,0.6],[0.6,0.5]])
                valid = 0
            sample['valid'].append(valid)
            sample['crop_polygon'].append(_crop_polygon)
        return sample

    def __str__(self):
        return 'CropFromMask:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'

class FixedResizeStretchMulticomp(object):
    def __init__(self, resolutions=None, flagvals=None):
        self.resolutions = resolutions
        self.flagvals = flagvals
        if self.flagvals is not None:
            assert(len(self.resolutions) == len(self.flagvals))

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample

        elems = list(sample.keys())
        for elem in elems:
            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if isinstance(sample[elem], list):
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(self.resolutions[elem], [3, len(sample[elem])])
                    else:
                        output_size = np.append(self.resolutions[elem], len(sample[elem]))
                    tmp = sample[elem]
                    sample[elem] = np.zeros(output_size, dtype=np.float32)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem])
                        else:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem], flagval=self.flagvals[elem])
                else:
                    if self.flagvals is None:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem])
                    else:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem], flagval=self.flagvals[elem])
        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)

class ToTensorStretchMulticomp(object):
    def __call__(self, sample):
        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue

            tmp = sample[elem]
            if 'polygon' in elem:
                sample[elem] = [torch.from_numpy(t).unsqueeze(0) for t in tmp]
                if 'sample' in elem or 'init' in elem:
                    sample[elem] = torch.cat(sample[elem], dim = 0)
                continue

            if elem == 'gt':
                tmp = [torch.from_numpy(t).unsqueeze(0).unsqueeze(0) for t in tmp]
                sample[elem] = tmp
                continue

            if elem == 'crop_image':
                tmp = tmp.transpose((3, 2, 0, 1))
                sample[elem] = torch.FloatTensor(tmp)
                continue

            if elem == 'image' or elem == 'crop_gt':
                tmp = tmp.transpose((2, 0, 1))
                sample[elem] = torch.FloatTensor(tmp).unsqueeze(0)
                continue

            if elem == 'valid':
                continue
            raise ValueError

        return sample

    def __str__(self):
        return 'ToTensor'

