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
from Evaluation import metrics
from functools import partial

def evaluate_instance(json_path, preds_dir):
    """
    Evaluate single instance.
    """
    with open(json_path, 'r') as f:
        ann = json.load(f)

    # Get label
    label = ann['label']
    gt_mask = io.imread(os.path.join(preds_dir, ann['gt_mask_fname']))
    pred_mask = io.imread(os.path.join(preds_dir, ann['pred_mask_fname']))

    # Get IOU
    iou = metrics.iou_from_mask(gt_mask, pred_mask)

    # Get number of corrections-
    n_corrections = [np.sum(ann['n_corrections'])]

    return label, iou, n_corrections

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
    ious = {k: [] for k in cityutils.LABEL_DICT.keys()}
    corrections = {k: [] for k in cityutils.LABEL_DICT.keys()}
    n_instances = {k: 0 for k in cityutils.LABEL_DICT.keys()}

    anns = glob(os.path.join(preds_dir, '*.json'))

    # Process annotations
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    print "Using %i cpus" % multiprocessing.cpu_count()

    evaluate_instance_fn = lambda x: evaluate_instance(x, preds_dir)

    results = []
    for x in tqdm(pool.imap_unordered(partial(evaluate_instance, preds_dir=preds_dir), anns), 
        total=len(anns)):

        if x[0] != 0:
            results.append(x)
    pool.close()
    pool.join()

    for result in tqdm(results):
        # Evaluate instance
        label, iou, n_corrections = result

        # Append results
        ious[label].append(iou)
        corrections[label] += n_corrections
        n_instances[label] += 1

    # Print results
    print_results_cityscapes(ious, corrections, n_instances)

    # Save JSON
    if output_path is not None:
        json_results_cityscapes(ious, corrections, n_instances, output_path)

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