import torch, cv2
import numpy as np

EPS = 1e-7

def tens2image(im):
    if im.size()[0] == 1:
        tmp = np.squeeze(im.numpy(), axis=0)
    else:
        tmp = im.numpy()
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))

def get_bbox(mask, points=None, pad=0, zero_pad=False):
    if points is not None:
        inds = np.flip(points.transpose(), axis=0)
    else:
        inds = np.where(mask > 0)

    if inds[0].shape[0] == 0:
        return None

    if zero_pad:
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf
    else:
        x_min_bound = 0
        y_min_bound = 0
        x_max_bound = mask.shape[1] - 1
        y_max_bound = mask.shape[0] - 1

    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[0].min() - pad, y_min_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[0].max() + pad, y_max_bound)

    return x_min, y_min, x_max, y_max

def crop_from_bbox(img, bbox, zero_pad=False, return_crop_bbox=False):
    bounds = (0, 0, img.shape[1] - 1, img.shape[0] - 1)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    if zero_pad:
        crop = np.zeros((bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), dtype=img.dtype)
        offsets = (-bbox[0], -bbox[1])

    else:
        assert (bbox == bbox_valid)
        crop = np.zeros((bbox_valid[3] - bbox_valid[1] + 1, bbox_valid[2] - bbox_valid[0] + 1), dtype=img.dtype)
        offsets = (-bbox_valid[0], -bbox_valid[1])

    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    img = np.squeeze(img)
    if img.ndim == 2:
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1]
    else:
        crop = np.tile(crop[:, :, np.newaxis], [1, 1, 3])  # Add 3 RGB Channels
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1, :]
    if return_crop_bbox:
        return crop, bbox_valid
    return crop

def crop_from_bbox_polygon(polygon, img, bbox, zero_pad=False):
    bounds = (0, 0, img.shape[1] - 1, img.shape[0] - 1)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))
    offsets = (-bbox[0], -bbox[1])

    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    new_polygon = np.zeros_like(polygon).astype(np.float)
    polygon = polygon.astype(np.float)
    new_polygon[:, 1] = (polygon[:, 1] - bbox_valid[1] + inds[1]) / float(bbox_valid[3] - bbox_valid[1] + 1)
    new_polygon[:, 0] = (polygon[:, 0] - bbox_valid[0] + inds[0]) / float(bbox_valid[2] - bbox_valid[0] + 1)
    new_polygon = np.clip(new_polygon, 0 + EPS, 1 - EPS)
    assert np.sum((new_polygon > 1).astype(np.float)) == 0
    assert np.sum((new_polygon < 0).astype(np.float)) == 0
    return new_polygon

def fixed_resize(sample, resolution, flagval=None):
    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[np.argmax(sample.shape[:2])] = int(
            round(float(resolution) / np.min(sample.shape[:2]) * np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
    return sample

def crop_from_mask(img, mask, relax=0, zero_pad=False):
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, dsize=tuple(reversed(img.shape[:2])), interpolation=cv2.INTER_NEAREST)
    assert (mask.shape[:2] == img.shape[:2])
    bbox = get_bbox(mask, pad=relax, zero_pad=zero_pad)
    if bbox is None:
        return None
    crop = crop_from_bbox(img, bbox, zero_pad)
    return crop

def crop_from_mask_polygon(poly, img, mask, relax=0, zero_pad=False):
    assert (mask.shape[:2] == img.shape[:2])
    bbox = get_bbox(mask, pad=relax, zero_pad=zero_pad)
    crop = crop_from_bbox_polygon(poly, img, bbox, zero_pad)
    return crop