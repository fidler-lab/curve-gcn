import os
import json
import argparse
import numpy as np
import skimage.io as io
from glob import glob
from tqdm import tqdm
import multiprocessing
import sys
sys.path.append('.')
from Utils import cityscapes_utils as cityutils
from functools import partial
import math

def db_eval_boundary(foreground_mask, gt_mask, bound_th=2):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))


    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask)
    gt_boundary = seg2bmap(gt_mask)



    from skimage.morphology import binary_dilation, disk

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall);

    return F


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
 """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), \
        'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1;

    return bmap


def evaluate_instance(json_path, preds_dir):
    """
    Evaluate single instance.
    """
    with open(json_path, 'r') as f:
        ann = json.load(f)

    # Get label
    label = ann['label']
    # #
    # if len(ann['components']) == 1:
    #     return 0, 0, 0
    # Get images`
    gt_mask = io.imread(os.path.join(preds_dir, ann['gt_mask_fname']))
    pred_mask = io.imread(os.path.join(preds_dir, ann['pred_mask_fname']))

    # Get IOU
    F = db_eval_boundary(pred_mask, gt_mask)

    # Get number of corrections
    n_corrections = [np.sum(ann['n_corrections'])]

    return label, n_corrections, F

def print_results_cityscapes(ious, corrections, n_instances):
    """
    Print results.
    """
    # Number of  Instances
    print
    print 'Number of instances'
    print '-' * 16
    for k, v in n_instances.iteritems():
        print '{}: {}'.format(k, v)

    # IOUS
    print '-' * 16
    print
    print 'IOUs'
    print '-' * 16
    means = []
    for k, v in ious.iteritems():
        mean = np.mean(v)
        means.append(mean)
        print '{}: MEAN: {} STD: {}'.format(k, mean, np.std(v))
    print 'ALL MEAN: {}'.format(np.mean(means))

    # N corrections
    print '-' * 16
    print
    print 'N corrections'
    print '-' * 16
    means = []
    for k, v in corrections.iteritems():
        if v:
            mean = np.mean(v)
            means.append(mean)
            print '{} MEAN: {} STD: {}'.format(k, mean, np.std(v))
        else:
            means.append(0.)
            print '{} MEAN: {} STD: {}'.format(k, 0, 0)
    print 'ALL MEAN: {}'.format(np.mean(means))
    print '-' * 16

def json_results_cityscapes(ious, corrections, n_instances, output_path):
    """
    Save results.
    """
    json_results = {}

    # N instances
    json_results['n_instances'] = {}
    for k, v in n_instances.iteritems():
        json_results['n_instances'][k] = v

    # IOUS
    json_results['ious'] = {}
    means = []
    for k, v in ious.iteritems():
        cur_result = {}
        if v:
            cur_result['mean'] = float(np.mean(v))
            cur_result['std'] = float(np.std(v))
        else:
            cur_result['mean'] = 0
            cur_result['std'] = 0

        means.append(cur_result['mean'])
        json_results['ious'][k] = cur_result
    json_results['ious']['all'] = np.mean(means)

    # CORRECTIONS
    json_results['corrections'] = {}
    for k, v in corrections.iteritems():
        cur_result = {}
        if v:
            cur_result['mean'] = float(np.mean(v))
            cur_result['std'] = float(np.std(v))
        else:
            cur_result['mean'] = 0
            cur_result['std'] = 0

        json_results['corrections'][k] = cur_result

    # Save results
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

def evaluate_cityscapes(preds_dir, split, output_path=None):
    """
    Read the new multi component json files.
    """
    # Counters
    F_scores = {k: [] for k in cityutils.LABEL_DICT.keys()}
    corrections = {k: [] for k in cityutils.LABEL_DICT.keys()}
    n_instances = {k: 0 for k in cityutils.LABEL_DICT.keys()}

    anns = glob(os.path.join(preds_dir, '*.json'))

    # Process annotations
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    print "Using %i cpus" % multiprocessing.cpu_count()


    results = []

    for x in tqdm(pool.imap_unordered(partial(evaluate_instance, preds_dir=preds_dir), anns),
        total=len(anns)):

        if x[0] != 0:
            results.append(x)
    pool.close()
    pool.join()


    # results = []
    #
    #
    # for x in anns:
    #     curr = evaluate_instance(x, preds_dir)
    #     if curr[0] != 0:
    #         results.append(curr)
    #

    for result in tqdm(results):
        # Evaluate instance
        label, n_corrections, F = result

        # Append results
        F_scores[label].append(F)
        corrections[label] += n_corrections
        n_instances[label] += 1



    # Print results
    print_results_cityscapes(F_scores, corrections, n_instances)

    # Save JSON
    if output_path is not None:
        json_results_cityscapes(F_scores, corrections, n_instances, output_path)

def main():
    """
    Main function.
    """
    # Command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pred',
        help='Predictions dir',
    )
    parser.add_argument(
        '--output',
        help='Path where to store the resuts',
        type=str,
        default=None
    )
    args = parser.parse_args()

    evaluate_cityscapes(args.pred, 'val', args.output)

if __name__ == '__main__':
    main()