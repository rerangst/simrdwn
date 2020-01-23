#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 06:14:20 2019

@author: avanetten

Parse COWC dataset for SIMRDWN training

Data located at:
    https://gdo152.llnl.gov/cowc/
    cd /raid/data/
    wget -r -np  ftp://gdo152.ucllnl.org/cowc/datasets/ground_truth_sets

"""

from __future__ import print_function
import shapely.geometry
import pandas as pd
import shapely
import pickle
import time
import cv2

from PyQt5 import QtCore

import os
import sys
import shutil
import importlib
import numpy as np

import simrdwn.data_prep.yolt_data_prep_funcs as yolt_data_prep_funcs
from simrdwn.core.thread_preprocess_tfrecords import Prep_TFrecordThread

from simrdwn.data_prep.data_conversion import *


class PrepDataThread(QtCore.QThread):

    sig1 = QtCore.pyqtSignal(str)
    # sig2 = QtCore.pyqtSignal(object)
    # sig3 = QtCore.pyqtSignal(list)

    def __init__(self):

        super().__init__()

        self.ground_truth_dir = ''
        self.train_out_dir = ''
        self.test_out_dir = ''
        self.label_map_path = ''
        self.cls_list_path = ''
        self.train_dirs = ['train']
        self.test_dirs = ['test']
        self.classes = []

        self.action = ''

        self.preprocess_tfrecords = Prep_TFrecordThread()
        self.data_conversion = DataConversionThread()
        # receive signal
        self.preprocess_tfrecords.sig1.connect(self.on_receive_signal)
        self.data_conversion.sig1.connect(self.on_receive_signal)
        return

    def update_args(self, ground_truth_dir='data/ground_truth_sets',
                    # label_map_path='data/train_data/class_labels.pbtxt',
                    train_out_dir='data/train_data',
                    test_out_dir='data/test_images',
                    verbose=True):
        self.verbose = verbose

        # label_path_root = 'data/train_data'
        self.ground_truth_dir = ground_truth_dir
        self.train_out_dir = train_out_dir
        self.test_out_dir = test_out_dir

        # self.label_map_path = label_map_path
        self.label_map_path = self.train_out_dir+'/class_labels.pbtxt'
        self.sig1.emit("label_map_path:" + self.label_map_path)
        
        self.cls_list_path = self.train_out_dir+'/classes.txt'

        ##############################
        # infer training output paths
        ##############################
        self.labels_dir = os.path.join(self.train_out_dir, 'labels_yolo/')
        self.images_dir = os.path.join(self.train_out_dir, 'images/')
        self.im_list_name = os.path.join(
            self.train_out_dir, 'yolt_train_list.txt')
        self.tfrecord_train = os.path.join(
            self.train_out_dir, 'train.tfrecord')
        self.sample_label_vis_dir = os.path.join(
            self.train_out_dir, 'sample_label_vis/')
        # im_locs_for_list = output_loc + train_name + '/' + 'training_data/images/'
        # train_images_list_file_loc = yolt_dir + 'data/'
        # create output dirs
        for d in [self.train_out_dir, self.test_out_dir, self.labels_dir, self.images_dir]:
            if not os.path.exists(d):
                self.sig1.emit("make dir:" + d)
                os.makedirs(d)

        if os.path.normpath(self.label_map_path) != os.path.join(self.train_out_dir, 'class_labels.pbtxt'):
            shutil.copy(self.label_map_path, os.path.join(
                self.train_out_dir, 'class_labels.pbtxt'))

        ##############################
        # set yolt training box size
        ##############################
        car_size = 3      # meters
        GSD = 0.15        # meters
        self.yolt_box_size = np.rint(car_size/GSD)  # size in pixels
        self.sig1.emit("self.yolt_box_size (pixels):" +
                       str(self.yolt_box_size))

        ##############################
        # slicing variables
        ##############################
        self.slice_overlap = 0.1
        self.zero_frac_thresh = 0.2
        self.sliceHeight, self.sliceWidth = 544, 544  # for for 82m windows

        return

    def run(self):
        #########################################
        # set yolt category params from pbtxt
        #########################################
        self.label_map_dict = self.preprocess_tfrecords.load_pbtxt(
            self.label_map_path, verbose=False)
        self.classes = self.label_map_dict.values()
        with open(self.cls_list_path, 'w') as f:
            f.write('\n'.join(self.classes))
        self.sig1.emit("self.label_map_dict:" + str(self.label_map_dict))
        # get ordered keys
        key_list = sorted(self.label_map_dict.keys())
        # category_num = len(key_list)
        # category list for yolt
        self.cat_list = [self.label_map_dict[k] for k in key_list]
        self.sig1.emit("cat list:" + str(self.cat_list))
        yolt_cat_str = ','.join(self.cat_list)
        self.sig1.emit("yolt cat str:" + yolt_cat_str)
        # create yolt_category dictionary (should start at 0, not 1!)
        self.yolt_cat_dict = {x: i for i, x in enumerate(self.cat_list)}
        self.sig1.emit("self.yolt_cat_dict:" + str(self.yolt_cat_dict))
        # conversion between yolt and pbtxt numbers (just increase number by 1)
        self.convert_dict = {x: x+1 for x in range(100)}
        self.sig1.emit("self.convert_dict:" + str(self.convert_dict))
        ##############################

        if self.action == "slice":
            self.slice_images()
        elif self.action == "label":
            self.label_sliced_images()
        elif self.action == "prepare_dataset":
            self.prepare_dataset()
        elif self.action == "convert_voc2yolo":
            self.convert_voc2yolo()

    def on_receive_signal(self, msg):
        self.sig1.emit(msg)

    def slice_images(self):
        ##############################
        # Slice large images into smaller chunks
        ##############################
        self.sig1.emit("self.im_list_name:" + self.im_list_name)
        if os.path.exists(self.im_list_name):
            run_slice = False
        else:
            run_slice = True

        for i, d in enumerate(self.train_dirs):
            dtot = os.path.join(self.ground_truth_dir, d)
            self.sig1.emit("dtot:" + dtot)

            # get label files
            files = os.listdir(dtot)
            img_files = [f for f in files if f.endswith(
                '.jpg') or f.endswith('.png')]

            for imfile in img_files:
                ext = imfile.split('.')[-1]
                name_root = imfile.split(ext)[0]
                imfile_tot = os.path.join(dtot, imfile)
                outroot = d + '_' + imfile.split('.'+ext)[0]

                self.sig1.emit("\nName_root" + name_root)
                self.sig1.emit("  imfile:" + imfile)
                self.sig1.emit("  imfile_tot:" + imfile_tot)
                self.sig1.emit("  outroot:" + outroot)

                if run_slice:
                    self.slice_im_no_mask(
                        imfile_tot, outroot, self.images_dir,
                        sliceHeight=self.sliceHeight, sliceWidth=self.sliceWidth,
                        zero_frac_thresh=self.zero_frac_thresh, overlap=self.slice_overlap,
                        pad=0, verbose=self.verbose)

        ##############################
        # Get list for simrdwn/data/, copy to data dir
        ##############################
        train_ims = [os.path.join(self.images_dir, f)
                     for f in os.listdir(self.images_dir)]
        f = open(self.im_list_name, 'w')
        for item in train_ims:
            f.write("%s\n" % item)
        f.close()


        #################################
        # Copy test images to test dir
        #################################
        self.sig1.emit("Copying test images to:" + self.test_out_dir)
        for td in self.test_dirs:
            td_tot_in = os.path.join(self.ground_truth_dir, td)
            # copy non-label files
            for f in os.listdir(td_tot_in):
                shutil.copy(os.path.join(td_tot_in, f), self.test_out_dir)
            # copy everything?
            #os.system('cp -r ' + td_tot + ' ' + self.test_out_dir)
            ##shutil.copytree(td_tot, self.test_out_dir)

        return

    def label_sliced_images(self):
        print('self.images_dir', self.images_dir)
        train_ims = [os.path.join(self.images_dir, f)
                     for f in os.listdir(self.images_dir)]
        f = open(self.im_list_name, 'w')
        for item in train_ims:
            f.write("%s\n" % item)
        f.close()

        return

    def prepare_dataset(self):

        #######################################################################
        # Ensure labels were created correctly by plotting a few
        #######################################################################
        max_plots = 50
        thickness = 2
        yolt_data_prep_funcs.plot_training_bboxes(
            self.labels_dir, self.images_dir, ignore_augment=False,
            sample_label_vis_dir=self.sample_label_vis_dir,
            max_plots=max_plots, thickness=thickness, ext='.png')

        ##############################
        # Make a .tfrecords file
        ##############################
        # importlib.reload(preprocess_tfrecords)
        self.preprocess_tfrecords.yolt_imlist_to_tf(image_list_file=self.im_list_name,
                                                    label_map_dict=self.label_map_dict,
                                                    TF_RecordPath=self.tfrecord_train,
                                                    TF_PathVal='', val_frac=0.0,
                                                    convert_dict=self.convert_dict, verbose=True)

        return

    def slice_im_no_mask(self, input_im, outname_root, outdir_im,
                         sliceHeight=256, sliceWidth=256,
                         zero_frac_thresh=0.2, overlap=0.2, pad=0, verbose=False):
        '''
        ADAPTED FROM YOLT/SCRIPTS/SLICE_IM.PY
        Assume input_im is rgb
        Slice large satellite image into smaller pieces,
        ignore slices with a percentage null greater then zero_fract_thresh'''

        ext = '.' + input_im.split('.')[-1]

        image = cv2.imread(input_im, 1)  # color
        self.sig1.emit("image.shape: "+str(image.shape))

        im_h, im_w = image.shape[:2]
        win_size = sliceHeight*sliceWidth

        # if slice sizes are large than image, pad the edges
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

        self.sig1.emit('overlap: '+str(overlap))
        self.sig1.emit('dx: '+str(dx))
        self.sig1.emit('dy: '+str(dy))
        self.sig1.emit('pad: '+str(pad))

        for y in range(0, im_h, dy):  # sliceHeight):
            for x in range(0, im_w, dx):  # sliceWidth):
                n_ims += 1

                if (n_ims % 50) == 0:
                    self.sig1.emit(str(n_ims))

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
                        self.sig1.emit(
                            "Zero frac too high at: "+str(zero_frac))
                    continue

                #  save
                outname_part = outname_root + '_' + \
                    '_' + str(y0) + '_' + str(x0) + \
                    '_' + str(sliceHeight) + '_' + str(sliceWidth) + \
                    '_' + str(pad) + \
                    '_' + str(im_w) + '_' + str(im_h)
                outname_im = os.path.join(outdir_im, outname_part + '.png')
                # txt_outpath = os.path.join(outdir_label, outname_part + '.txt')

                # save yolt ims
                if verbose:
                    self.sig1.emit("image output: "+outname_im)
                cv2.imwrite(outname_im, window_c)

                n_ims_nonull += 1

        self.sig1.emit("Num slices: "+str(n_ims)+", Num non-null slices: "+str(n_ims_nonull) +
                       ", sliceHeight: "+str(sliceHeight)+", sliceWidth: "+str(sliceWidth))
        self.sig1.emit("Time to slice "+input_im+": " +
                       str(time.time()-t0)+" seconds")

        return


    def convert_voc2yolo(self):
        voc_root = os.path.join(self.train_out_dir, 'labels_voc')
        self.sig1.emit('Labels dir (voc format): '+voc_root)
        self.sig1.emit('Labels dir (yolo format): '+self.labels_dir)
        
        voc = VOC(self.data_conversion)
        yolo = YOLO(os.path.abspath(self.cls_list_path), self.data_conversion)

        flag, data = voc.parse(voc_root)

        if flag == True:

            flag, data = yolo.generate(data)
            if flag == True:
                flag, data = yolo.save(data, self.labels_dir, self.images_dir , '.png')
                if flag == False:
                    self.sig1.emit("Saving Result : {}, msg : {}".format(flag, data))
            else:
                self.sig1.emit("YOLO Generating Result : {}, msg : {}".format(flag, data))
        else:
            self.sig1.emit("VOC Parsing Result : {}, msg : {}".format(flag, data))
