# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 02:53:01 2016

@author: avanetten
"""

from __future__ import print_function
import os
import cv2
import time
import numpy as np
import skimage.io


###############################################################################
def slice_im(image_path, out_name, out_dir, sliceHeight=256, sliceWidth=256,
             zero_frac_thresh=0.2, overlap=0.2, slice_sep='|',
             out_ext='.png', verbose=False):

    """
    Slice a large image into smaller windows

    Arguments
    ---------
    image_path : str
        Location of image to slice
    out_name : str
        Root name of output files (coordinates will be appended to this)
    out_dir : str
        Output directory
    sliceHeight : int
        Height of each slice.  Defaults to ``256``.
    sliceWidth : int
        Width of each slice.  Defaults to ``256``.
    zero_frac_thresh : float
        Maximum fraction of window that is allowed to be zeros or null.
        Defaults to ``0.2``.
    overlap : float
        Fractional overlap of each window (e.g. an overlap of 0.2 for a window
        of size 256 yields an overlap of 51 pixels).
        Default to ``0.2``.
    slice_sep : str
        Character used to separate outname from coordinates in the saved
        windows.  Defaults to ``|``
    out_ext : str
        Extension of saved images.  Defaults to ``.png``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``

    Returns
    -------
    None
    """

    if len(out_ext) == 0:
        ext = '.' + image_path.split('.')[-1]
    else:
        ext = out_ext

    image = cv2.imread(image_path, 1)  # color
    print("image.shape:", image.shape)

    im_h, im_w = image.shape[:2]
    win_size = sliceHeight*sliceWidth

    # if slice sizes are large than image, pad the edges
    pad = 0
    if sliceHeight > im_h:
        pad = sliceHeight - im_h
    if sliceWidth > im_w:
        pad = max(pad, sliceWidth - im_w)
    # pad the edge of the image with black pixels
    if pad > 0:
        border_color = (0, 0, 0)
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                   cv2.BORDER_CONSTANT, value=border_color)

    t0 = time.time()
    n_ims = 0
    n_ims_nonull = 0
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)

    print('overlap', overlap)
    print('dx', dx)
    print('dy', dy)
    print('pad', pad)

    for y in range(0, im_h, dy):  # sliceHeight):
        for x in range(0, im_w, dx):  # sliceWidth):
            n_ims += 1

            if (n_ims % 50) == 0:
                print(n_ims)

            # extract image
            # make sure we don't go past the edge of the image
            if y + sliceHeight > im_h:
                y0 = im_h - sliceHeight
            else:
                y0 = y
            if x + sliceWidth > im_w:
                x0 = im_w - sliceWidth
            else:
                x0 = x

            window_c = image[y0:y0 + sliceHeight, x0:x0 + sliceWidth]
            win_h, win_w = window_c.shape[:2]

            # get black and white image
            window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)

            # find threshold of image that's not black
            # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html?highlight=threshold
            ret, thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
            non_zero_counts = cv2.countNonZero(thresh1)
            zero_counts = win_size - non_zero_counts
            zero_frac = float(zero_counts) / win_size
            # print ("zero_frac", zero_fra
            # skip if image is mostly empty
            if zero_frac >= zero_frac_thresh:
                if verbose:
                    print("Zero frac too high at:", zero_frac)
                continue

            #  save
            outpath = os.path.join(
                    out_dir,
                    out_name + slice_sep + str(y0) + '_' + str(x0)
                    + '_' + str(sliceHeight) + '_' + str(sliceWidth)
                    + '_' + str(pad)
                    + '_' + str(im_w) + '_' + str(im_h)
                    + ext)

            cv2.imwrite(outpath, window_c)

            n_ims_nonull += 1

    print("Num slices:", n_ims, "Num non-null slices:", n_ims_nonull,
          "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
    print("Time to slice", image_path, time.time()-t0, "seconds")

    return
