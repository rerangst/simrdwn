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


import os
import sys
import shutil
import importlib
import numpy as np

import cv2

# import simrdwn.data_prep.parse_cowc as parse_cowc
# import simrdwn.data_prep.yolt_data_prep_funcs as yolt_data_prep_funcs

import parse_cowc
import yolt_data_prep_funcs

class PrepData:
    def __init__(self, cowc_data_dir='data/ground_truth_sets',
                 label_map_file='class_labels_car.pbtxt',
                 simrdwn_data_dir='data/train_data',
                 train_out_dir='data/train_data/cowc',
                 test_out_dir='data/test_images/cowc',
                 verbose=True):

        return
    
    def update_args(self, cowc_data_dir='data/ground_truth_sets',
                    label_map_file='class_labels_car.pbtxt',
                    simrdwn_data_dir='data/train_data',
                    train_out_dir='data/train_data/cowc',
                    test_out_dir='data/test_images/cowc',
                    verbose=True):
        # self.path_simrdwn_utils = os.getcwd()
        self.path_simrdwn_utils = os.path.dirname(os.path.realpath(__file__))

        ###############################################################################
        # path variables (may need to be edited! )

        # gpu07
        # label_map_file = 'class_labels_car.pbtxt'
        self.verbose = verbose

        # at /cosmiq
        self.simrdwn_data_dir = simrdwn_data_dir
        # label_path_root = 'data/train_data'
        self.train_out_dir = train_out_dir
        self.test_out_dir = test_out_dir
        # at /local_data
        # self.simrdwn_data_dir = '/local_data/simrdwn3/data/train_data'
        # label_path_root = '/local_data/simrdwn3/data/train_data'
        # self.test_out_dir = '/local_data/simrdwn3/data/test_images/cowc'

        # self.label_map_path = os.path.join(label_path_root, label_map_file)
        self.label_map_path = os.path.join(
            self.simrdwn_data_dir, label_map_file)
        print("self.label_map_path:", self.label_map_path)

        ##############################
        # list of train and test directories
        # for now skip Columbus and Vahingen since they are grayscale
        # os.path.join(args.cowc_data_dir, 'datasets/ground_truth_sets/')
        self.ground_truth_dir = cowc_data_dir
        self.train_dirs = ['train']
        self.test_dirs = ['test']
        # self.train_dirs = ['Potsdam_ISPRS', 'Selwyn_LINZ', 'Toronto_ISPRS']
        # self.test_dirs = ['Utah_AGRC']
        self.annotation_suffix = '_Annotated_Cars.png'
        ##############################

        ##############################
        # infer training output paths
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
                print("make dir:", d)
                os.makedirs(d)
        ##############################

        ##############################
        # set yolt training box size
        car_size = 3      # meters
        GSD = 0.15        # meters
        self.yolt_box_size = np.rint(car_size/GSD)  # size in pixels
        print("self.yolt_box_size (pixels):", self.yolt_box_size)
        ##############################

        ##############################
        # slicing variables
        self.slice_overlap = 0.1
        self.zero_frac_thresh = 0.2
        self.sliceHeight, self.sliceWidth = 544, 544  # for for 82m windows
        ##############################

    def gen_data(self):
        sys.path.append(os.path.join(self.path_simrdwn_utils, '..', 'core'))
        import preprocess_tfrecords
        ##############################
        # set yolt category params from pbtxt
        label_map_dict = preprocess_tfrecords.load_pbtxt(
            self.label_map_path, verbose=False)
        print("label_map_dict:", label_map_dict)
        # get ordered keys
        key_list = sorted(label_map_dict.keys())
        category_num = len(key_list)
        # category list for yolt
        cat_list = [label_map_dict[k] for k in key_list]
        print("cat list:", cat_list)
        yolt_cat_str = ','.join(cat_list)
        print("yolt cat str:", yolt_cat_str)
        # create yolt_category dictionary (should start at 0, not 1!)
        yolt_cat_dict = {x: i for i, x in enumerate(cat_list)}
        print("yolt_cat_dict:", yolt_cat_dict)
        # conversion between yolt and pbtxt numbers (just increase number by 1)
        convert_dict = {x: x+1 for x in range(100)}
        print("convert_dict:", convert_dict)
        ##############################

        ##############################
        # Slice large images into smaller chunks
        ##############################
        print("self.im_list_name:", self.im_list_name)
        if os.path.exists(self.im_list_name):
            run_slice = False
        else:
            run_slice = True

        for i, d in enumerate(self.train_dirs):
            dtot = os.path.join(self.ground_truth_dir, d)
            print("dtot:", dtot)

            # get label files
            files = os.listdir(dtot)
            annotate_files = [
                f for f in files if f.endswith(self.annotation_suffix)]
            # print ("annotate_files:", annotate_files
            img_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]

            for imfile in img_files:
                ext = imfile.split('.')[-1]
                name_root = imfile.split(ext)[0]
                annotate_file = name_root + self.annotation_suffix
                annotate_file_tot = os.path.join(dtot, annotate_file)
                imfile_tot = os.path.join(dtot, imfile)
                outroot = d + '_' + imfile.split('.'+ext)[0]
                print("\nName_root", name_root)
                print("   annotate_file:", annotate_file)
                print("  imfile:", imfile)
                print("  imfile_tot:", imfile_tot)
                print("  outroot:", outroot)

                if run_slice:
                    if os.path.exists(annotate_file_tot):
                        parse_cowc.slice_im_cowc(
                            imfile_tot, annotate_file_tot, outroot,
                            self.images_dir, self.labels_dir, yolt_cat_dict, cat_list[0],
                            self.yolt_box_size, sliceHeight=self.sliceHeight, sliceWidth=self.sliceWidth,
                            zero_frac_thresh=self.zero_frac_thresh, overlap=self.slice_overlap,
                            pad=0, verbose=self.verbose)
                    else:
                        parse_cowc.slice_im_no_mask(
                            imfile_tot, outroot,
                            self.images_dir, self.labels_dir, yolt_cat_dict, cat_list[0],
                            self.yolt_box_size, sliceHeight=self.sliceHeight, sliceWidth=self.sliceWidth,
                            zero_frac_thresh=self.zero_frac_thresh, overlap=self.slice_overlap,
                            pad=0, verbose=self.verbose)
        ##############################

        ##############################
        # Get list for simrdwn/data/, copy to data dir
        ##############################
        train_ims = [os.path.join(self.images_dir, f)
                     for f in os.listdir(self.images_dir)]
        f = open(self.im_list_name, 'w')
        for item in train_ims:
            f.write("%s\n" % item)
        f.close()
        # copy to data dir
        print("Copying", self.im_list_name, "to:", self.simrdwn_data_dir)
        shutil.copy(self.im_list_name, self.simrdwn_data_dir)
        ##############################

        ##############################
        # Ensure labels were created correctly by plotting a few
        ##############################
        max_plots = 50
        thickness = 2
        yolt_data_prep_funcs.plot_training_bboxes(
            self.labels_dir, self.images_dir, ignore_augment=False,
            sample_label_vis_dir=self.sample_label_vis_dir,
            max_plots=max_plots, thickness=thickness, ext='.png', verbose=True)

        ##############################
        # Make a .tfrecords file
        ##############################
        importlib.reload(preprocess_tfrecords)
        preprocess_tfrecords.yolt_imlist_to_tf(self.im_list_name, label_map_dict,
                                               TF_RecordPath=self.tfrecord_train,
                                               TF_PathVal='', val_frac=0.0,
                                               convert_dict=convert_dict, verbose=True)
        # copy train file to data dir
        print("Copying", self.tfrecord_train, "to:", self.simrdwn_data_dir)
        shutil.copy(self.tfrecord_train, self.simrdwn_data_dir)

        ##############################
        # Copy test images to test dir
        print("Copying test images to:", self.test_out_dir)
        for td in self.test_dirs:
            td_tot_in = os.path.join(self.ground_truth_dir, td)
            td_tot_out = os.path.join(self.test_out_dir, td)
            if not os.path.exists(td_tot_out):
                os.makedirs(td_tot_out)
            # copy non-label files
            for f in os.listdir(td_tot_in):
                if (f.endswith('.png') or f.endswith('.jpg')) and not f.endswith(('_Cars.png', '_Negatives.png', '.xcf')):
                    shutil.copy2(os.path.join(td_tot_in, f), td_tot_out)
            # copy everything?
            #os.system('cp -r ' + td_tot + ' ' + self.test_out_dir)
            ##shutil.copytree(td_tot, self.test_out_dir)
        ##############################


if __name__ == "__main__":

    prep_data = PrepData()
    prep_data.update_args(cowc_data_dir = '/media/tunguyen/DEVS/DeepLearning/simrdwn/data/tutu',
                          simrdwn_data_dir='/media/tunguyen/DEVS/DeepLearning/simrdwn/data/train_data_tutu',
                          label_map_file='/media/tunguyen/DEVS/DeepLearning/simrdwn/data/train_data_tutu/class_labels.pbtxt',
                          train_out_dir='/media/tunguyen/DEVS/DeepLearning/simrdwn/data/train_data_tutu/tutu',
                          test_out_dir='/media/tunguyen/DEVS/DeepLearning/simrdwn/data/test_images_tutu/tutu'
)
    prep_data.gen_data()
