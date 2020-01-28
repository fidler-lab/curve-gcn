from skimage.io import imread
import skimage.color as color
import cv2
import os
import numpy as np

def create_folder(path):
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s'%(path))
        print('Experiment folder created at: %s'%(path))

def rgb_img_read(img_path):
    """
    Read image and always return it as a RGB image (3D vector with 3 channels).
    """
    img = imread(img_path)
    if len(img.shape) == 2:
        img = color.gray2rgb(img)

    # Deal with RGBA
    img = img[..., :3]

    if img.dtype == 'uint8':
        # [0,1] image
        img = img.astype(np.float32)/255
    return img

def get_full_mask_from_instance(min_area, instance):
    img_h, img_w = instance['img_height'], instance['img_width']
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for component in instance['components']:
        p = np.array(component['poly'], np.int)
        if component['area'] < min_area:
            continue
        else:
            draw_poly(mask, p)
    return mask

def draw_poly(mask, poly):
    """
    NOTE: Numpy function
    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)
    cv2.fillPoly(mask, [poly], 255)
    return mask