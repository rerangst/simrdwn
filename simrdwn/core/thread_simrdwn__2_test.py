#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:11:56 2016

@author: avanetten

"""

from __future__ import print_function
import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
# import argparse
import shutil
import copy
# import logging
# import tensorflow as tf
import skimage.io
import cv2

from argparse import Namespace

import simrdwn.core.utils as utils
import simrdwn.core.post_process as post_process
import simrdwn.core.add_geo_coords as add_geo_coords
import simrdwn.core.thread_parse_tfrecord as parse_tfrecord
# import simrdwn.core.thread_parse_tfrecord
from simrdwn.core.thread_preprocess_tfrecords import Prep_TFrecordThread
# import simrdwn.core.preprocess_tfrecords as preprocess_tfrecords

from PyQt5 import QtGui, QtCore

sys.stdout.flush()
##########################


class SimrdwnThread(QtCore.QThread):

    sig1 = QtCore.pyqtSignal(str)
    outputs = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.preprocess_tfrecords = Prep_TFrecordThread()

        # Construct argument parser
        self.args = {}

        # general settings
        self.args['framework'] = 'yolt2'
        self.args['mode'] = 'train'
        self.args['gpu'] = 0
        self.args['single_gpu_machine'] = 0
        self.args['nbands'] = 3
        self.args['outname'] = 'tmp'
        self.args['label_map_path'] = ''
        self.args['weight_file'] = 'yolo.weights'
        self.args['append_date_string'] = 1

        # training settings
        self.args['train_data_dir'] = ''
        self.args['yolt_train_images_list_file'] = ''
        self.args['max_batches'] = 60000
        self.args['batch_size'] = 32
        self.args['train_input_width'] = 416
        self.args['train_input_height'] = 416
        # TF api specific settings
        self.args['tf_cfg_train_file'] = ''
        self.args['train_tf_record'] = ''
        self.args['train_val_tf_record'] = ''

        # yolt specific
        # yolt_object_labels_str is now redundant, and only label_map_path is needed
        self.args['yolt_object_labels_str'] = ''

        # test settings
        self.args['train_model_path'] = ''
        self.args['use_tfrecords'] = 0
        self.args['test_presliced_tfrecord_path'] = ''
        self.args['test_presliced_list'] = ''
        self.args['testims_dir'] = ''
        self.args['slice_sizes_str'] = '416'
        self.args['edge_buffer_test'] = -1000
        self.args['max_edge_aspect_ratio'] = 3
        self.args['slice_overlap'] = 0.35
        self.args['nms_overlap_thresh'] = 0.5
        self.args['test_box_rescale_frac'] = 1.0
        self.args['test_slice_sep'] = '__'
        self.args['val_df_root_init'] = 'test_predictions_init.csv'
        self.args['val_df_root_aug'] = 'test_predictions_aug.csv'
        self.args['test_splitims_locs_file_root'] = 'test_splitims_input_files.txt'
        self.args['test_prep_only'] = 0
        self.args['overwrite_inference_graph'] = 0
        self.args['min_retain_prob'] = 0.025
        self.args['test_add_geo_coords'] = 1

        # test, specific to YOLT
        self.args['yolt_nms_thresh'] = 0.0
        # test plotting
        self.args['plot_thresh_str'] = '0.3'
        self.args['show_labels'] = 0
        self.args['alpha_scaling'] = 0
        self.args['show_test_plots'] = 0
        self.args['save_json'] = 1

        # self.args['plot_names']=0,
        #                    help']="Switch to show plots names in test")
        self.args['rotate_boxes'] = 0
        self.args['plot_line_thickness'] = 2
        self.args['n_test_output_plots'] = 10
        self.args['test_make_legend_and_title'] = 1
        self.args['test_im_compression_level'] = 6
        self.args['keep_test_slices'] = 0
        self.args['shuffle_val_output_plot_ims'] = 0

        # random YOLT specific settings
        self.args['yolt_cfg_file'] = 'yolo.cfg'
        self.args['subdivisions'] = 16
        self.args['use_opencv'] = 1
        self.args['boxes_per_grid'] = 5

        # if evaluating spacenet data
        self.args['building_csv_file'] = ''

        # second test classifier
        self.args['train_model_path2'] = ''
        self.args['label_map_path2'] = ''
        self.args['weight_file2'] = ''
        self.args['slice_sizes_str2'] = '0'
        self.args['plot_thresh_str2'] = '0.3'
        self.args['inference_graph_path2'] = 'outputs/ssd/output_inference_graph/frozen_inference_graph.pb'
        self.args['yolt_cfg_file2'] = 'yolo.cfg'
        self.args['val_df_root_init2'] = 'test_predictions_init2.csv'
        self.args['val_df_root_aug2'] = 'test_predictions_aug2.csv'
        self.args['test_splitims_locs_file_root2'] = 'test_splitims_input_files2.txt'
        # self.args['test_prediction_pkl_root2']='val_refine_preds2.pkl',
        #                   help']="Root of test pickle")

        # total test
        self.args['val_df_root_tot'] = 'test_predictions_tot.csv'
        self.args['val_prediction_df_refine_tot_root_part'] = 'test_predictions_refine'

        # Defaults that rarely should need changed
        self.args['multi_band_delim'] = '#'
        self.args['zero_frac_thresh'] = 0.5
        self.args['str_delim'] = ','

        self.args = Namespace(**self.args)

        return

    ###############################################################################
    def update_args(self):
        """
        Update self.args (mostly paths)

        Arguments
        ---------
        self.args : argparse
            Input self.args passed to simrdwn

        Returns
        -------
        self.args : argparse
            Updated self.args
        """

        ###########################################################################
        # CONSTRUCT INFERRED VALUES
        ###########################################################################

        ##########################
        # GLOBAL VALUES
        # set directory structure

        # self.args.src_dir = os.path.dirname(os.path.realpath(__file__))
        self.args.core_dir = os.path.dirname(os.path.realpath(__file__))
        self.args.this_file = os.path.join(self.args.core_dir, 'simrdwn.py')
        self.args.simrdwn_dir = os.path.dirname(
            os.path.dirname(self.args.core_dir))
        self.args.results_topdir = os.path.join(
            self.args.simrdwn_dir, 'results')
        self.args.tf_cfg_dir = os.path.join('tf', 'cfg')
        self.args.yolt_plot_file = os.path.join(
            self.args.core_dir, 'yolt_plot_loss.py')
        self.args.tf_plot_file = os.path.join(
            self.args.core_dir, 'tf_plot_loss.py')

        # # if train_data_dir is not a full directory, set it as within simrdwn
        # if self.args.train_data_dir.startswith('/'):
        #     pass
        # else:
        #     self.args.train_data_dir = os.path.join(self.args.simrdwn_dir, self.args.train_data_dir)

        # # keep raw testims dir if it starts with a '/'
        # if self.args.testims_dir.startswith('/'):
        #     self.args.testims_dir_tot = self.args.testims_dir
        # else:
        #     # self.args.testims_dir_tot = os.path.join(self.args.simrdwn_dir,
        #     #                                     'data/test_images',
        #     #                                     self.args.testims_dir)

        #     self.args.testims_dir_tot = os.path.join(self.args.simrdwn_dir,
        #                                         self.args.testims_dir)
        # self.args.testims_dir_tot = self.args.testims_dir

        # print ("os.listdir(self.args.testims_dir_tot:",
        #   os.listdir(self.args.testims_dir_tot))
        # ensure test ims exist
        if (self.args.mode.upper() == 'TEST') and (not os.path.exists(self.args.testims_dir_tot)):
            raise ValueError("Test images directory does not exist: "
                             "{}".format(self.args.testims_dir_tot))

        if self.args.framework.upper().startswith('YOLT'):
            self.args.yolt_dir = os.path.join(
                self.args.simrdwn_dir, self.args.framework)
        else:
            self.args.yolt_dir = os.path.join(self.args.simrdwn_dir, 'yolt')

        # self.args.yolt_weight_dir = os.path.join(self.args.yolt_dir, 'input_weights')
        # self.args.yolt_cfg_dir = os.path.join(self.args.yolt_dir, 'cfg')

        self.sig1.emit('self.args.yolt_cfg_dir' + self.args.yolt_cfg_dir)

        ##########################################
        # Get datetime and set outlog file
        self.args.now = datetime.datetime.now()
        if bool(self.args.append_date_string):
            self.args.date_string = self.args.now.strftime('%Y_%m_%d_%H-%M-%S')
            # print "Date string:", date_string
            self.args.res_name = self.args.mode + '_' + self.args.framework + \
                '_' + self.args.outname + '_' + self.args.date_string
        else:
            self.args.date_string = ''
            self.args.res_name = self.args.mode + '_' + \
                self.args.framework + '_' + self.args.outname

        self.args.results_dir = os.path.join(
            self.args.results_topdir, self.args.res_name)
        self.args.log_dir = os.path.join(self.args.results_dir, 'logs')
        self.args.log_file = os.path.join(
            self.args.log_dir, self.args.res_name + '.log')
        self.args.yolt_loss_file = os.path.join(
            self.args.log_dir, 'yolt_loss.txt')
        self.args.labels_log_file = os.path.join(
            self.args.log_dir, 'labels_list.txt')

        # set total location of test image file list
        self.args.test_presliced_list_tot = os.path.join(
            self.args.results_topdir, self.args.test_presliced_list)
        # self.args.test_presliced_list_tot = os.path.join(self.args.simrdwn_dir,
        #   self.args.test_presliced_list)
        if len(self.args.test_presliced_tfrecord_path) > 0:
            self.args.test_presliced_tfrecord_tot = os.path.join(
                self.args.results_topdir, self.args.test_presliced_tfrecord_path,
                'test_splitims.tfrecord')
            self.args.test_tfrecord_out = os.path.join(
                self.args.results_dir, 'predictions.tfrecord')
        else:
            self.args.test_presliced_tfrecord_tot = ''
            self.args.test_tfrecord_out = ''

        if len(self.args.test_presliced_list) > 0:
            self.args.test_splitims_locs_file = self.args.test_presliced_list_tot
        else:
            self.args.test_splitims_locs_file = os.path.join(
                self.args.results_dir, self.args.test_splitims_locs_file_root)
        # self.args.test_tfrecord_file = os.path.join(self.args.results_dir,
        #   self.args.test_tfrecord_root)
        # self.args.val_prediction_pkl = os.path.join(self.args.results_dir,
        #   self.args.test_prediction_pkl_root)
        # self.args.val_df_tfrecords_out = os.path.join(self.args.results_dir,
        #   'predictions.tfrecord')
        self.args.val_df_path_init = os.path.join(
            self.args.results_dir, self.args.val_df_root_init)
        self.args.val_df_path_aug = os.path.join(
            self.args.results_dir, self.args.val_df_root_aug)

        self.args.inference_graph_path_tot = os.path.join(
            self.args.results_topdir, self.args.train_model_path,
            'frozen_model/frozen_inference_graph.pb')

        # and yolt cfg file
        self.args.yolt_cfg_file_tot = os.path.join(
            self.args.log_dir, self.args.yolt_cfg_file)

        # weight and cfg files
        # TRAIN
        if self.args.mode.upper() == 'TRAIN':
            self.args.weight_file_tot = os.path.join(
                self.args.yolt_weight_dir, self.args.weight_file)
            # assume weights are in weight_dir, and cfg in cfg_dir
            # self.args.yolt_cfg_file_in = os.path.join(
            #     self.args.yolt_cfg_dir, self.args.yolt_cfg_file)
            self.args.tf_cfg_train_file = os.path.join(
                self.args.tf_cfg_dir, self.args.tf_cfg_train_file)
            # self.args.yolt_cfg_file_in = self.args.yolt_cfg_file
            # self.args.weight_file_tot = self.args.weight_file
        # TEST
        # else:
        #     self.args.weight_file_tot = os.path.join(
        #         self.args.results_topdir, self.args.train_model_path, self.args.weight_file)
        #     self.args.tf_cfg_train_file = os.path.join(
        #         self.args.results_topdir, self.args.train_model_path,  # 'logs',
        #         self.args.tf_cfg_train_file)

        #     # assume weights and cfg are in the training dir
        #     self.args.yolt_cfg_file_in = os.path.join(os.path.dirname(
        #         self.args.weight_file_tot), 'logs/', self.args.yolt_cfg_file)

        # set training files (assume files are in train_data_dir unless a full
        #  path is given)
        self.args.yolt_train_images_list_file_tot = os.path.join(
            self.args.train_data_dir, 'yolt_train_list.txt')
        if not os.path.exists(self.args.yolt_train_images_list_file_tot) and self.args.mode.upper() == 'TRAIN':
            self.sig1.emit('yolt_train_images_list_file ' +
                           self.args.yolt_train_images_list_file_tot+' not found')
            return

        # train tf record
        if self.args.train_tf_record.startswith('/'):
            pass
        else:
            self.args.train_tf_record = os.path.join(
                self.args.train_data_dir, self.args.train_tf_record)

        ##########################
        # set tf cfg file out
        tf_cfg_base = os.path.basename(self.args.tf_cfg_train_file)
        # tf_cfg_root = tf_cfg_base.split('.')[0]
        self.args.tf_cfg_train_file_out = os.path.join(
            self.args.log_dir, tf_cfg_base)
        self.args.tf_model_output_directory = os.path.join(
            self.args.results_dir, 'frozen_model')
        # self.args.tf_model_output_directory = os.path.join(self.args.results_dir,
        #   tf_cfg_root)

        ##########################
        # set possible extensions for image files
        self.args.extension_list = ['.png', '.tif', '.TIF', '.TIFF', '.tiff', '.JPG',
                                    '.jpg', '.JPEG', '.jpeg']

        # self.args.test_make_pngs = bool(self.args.test_make_pngs)
        self.args.test_make_legend_and_title = bool(
            self.args.test_make_legend_and_title)
        self.args.keep_test_slices = bool(self.args.keep_test_slices)
        self.args.test_add_geo_coords = bool(self.args.test_add_geo_coords)

        # set cuda values
        if self.args.gpu >= 0:
            self.args.use_GPU, self.args.use_CUDNN = 1, 1
        else:
            self.args.use_GPU, self.args.use_CUDNN = 0, 0

        # update label_map_path, if needed
        if self.args.mode.upper() == 'TRAIN':
            self.args.label_map_path = os.path.join(
                self.args.train_data_dir, 'class_labels.pbtxt')

        if not os.path.exists(self.args.label_map_path):
            self.sig1.emit('label_map_path ' +
                           self.args.label_map_path+' not found')
            return

        # make label_map_dic (key=int, value=str), and reverse
        if len(self.args.label_map_path) > 0:
            self.args.label_map_dict = self.preprocess_tfrecords.load_pbtxt(
                self.args.label_map_path, verbose=False)
            # ensure dict is 1-indexed
            if min(list(self.args.label_map_dict.keys())) != 1:
                self.sig1.emit("Error: label_map_dict (" + self.args.label_map_path + ") must"
                               " be 1-indexed")
                return
        else:
            self.args.label_map_dict = {}

        # retersed labels
        self.args.label_map_dict_rev = {
            v: k for k, v in self.args.label_map_dict.items()}
        # self.args.label_map_dict_rev = {v: k for k,v
        #   in self.args.label_map_dict.iteritems()}
        # print ("label_map_dict:", self.args.label_map_dict)

        # infer lists from self.args
        if len(self.args.yolt_object_labels_str) == 0:
            self.args.yolt_object_labels = [self.args.label_map_dict[ktmp] for ktmp in
                                            sorted(self.args.label_map_dict.keys())]
            self.args.yolt_object_labels_str = ','.join(
                self.args.yolt_object_labels)
        else:
            self.args.yolt_object_labels = self.args.yolt_object_labels_str.split(
                ',')
            # also set label_map_dict, if it's empty
            if len(self.args.label_map_path) == 0:
                for itmp, val in enumerate(self.args.yolt_object_labels):
                    self.args.label_map_dict[itmp] = val
                self.args.label_map_dict_rev = {v: k for k,
                                                v in self.args.label_map_dict.items()}
                # self.args.label_map_dict_rev = {v: k for k,v in
                #   self.args.label_map_dict.iteritems()}

        # set total dict
        self.args.label_map_dict_tot = copy.deepcopy(self.args.label_map_dict)
        self.args.label_map_dict_rev_tot = copy.deepcopy(
            self.args.label_map_dict_rev)

        self.args.yolt_classnum = len(self.args.yolt_object_labels)

        # for yolov2
        self.args.yolt_final_output = 1 * 1 * \
            self.args.boxes_per_grid * (self.args.yolt_classnum + 4 + 1)
        # for yolov3
        # make sure num boxes is divisible by 3
        if self.args.framework.upper() == 'YOLT3' and self.args.boxes_per_grid % 3 != 0:
            self.sig1.emit("for YOLT3, boxes_per_grid must be divisble by 3!")
            self.sig1.emit("RETURNING!")
            return
        self.args.yolov3_filters = int(self.args.boxes_per_grid /
                                       3 * (self.args.yolt_classnum + 4 + 1))

        # plot thresh and slice sizes
        self.args.plot_thresh = np.array(
            self.args.plot_thresh_str.split(self.args.str_delim)).astype(float)
        self.args.slice_sizes = np.array(
            self.args.slice_sizes_str.split(self.args.str_delim)).astype(int)

        # set test list
        try:
            if self.args.nbands == 3:
                # print ("os.listdir(self.args.testims_dir_tot:",
                #   os.listdir(self.args.testims_dir_tot))
                self.args.test_ims_list = [f for f in os.listdir(self.args.testims_dir_tot)
                                           if f.endswith(tuple(self.args.extension_list))]
                # self.sig1.emit("self.args.test_ims_list:", self.args.test_ims_list)
            else:
                self.args.test_ims_list = [f for f in os.listdir(self.args.testims_dir_tot)
                                           if f.endswith('#1.png')]
        except:
            self.args.test_ims_list = []
        # print ("test_ims_list:", self.args.test_ims_list)
        # more test files
        self.args.rotate_boxes = bool(self.args.rotate_boxes)
        self.args.yolt_test_classes_files = [os.path.join(self.args.results_dir, l + '.txt')
                                             for l in self.args.yolt_object_labels]

        ##########################
        # get second test classifier values
        self.args.slice_sizes2 = []
        if len(self.args.label_map_path2) > 0:

            # label dict
            self.args.label_map_dict2 = self.preprocess_tfrecords.load_pbtxt(
                self.args.label_map_path2, verbose=False)
            self.args.label_map_dict_rev2 = {v: k for k,
                                             v in self.args.label_map_dict2.items()}
            # self.args.label_map_dict_rev2 = {v: k for k,v in
            #   self.args.label_map_dict2.iteritems()}

            # to update label_map_dict just adds second classifier to first
            nmax_tmp = max(self.args.label_map_dict.keys())
            for ktmp, vtmp in self.args.label_map_dict2.items():
                # for ktmp, vtmp in self.args.label_map_dict2.iteritems():
                self.args.label_map_dict_tot[ktmp+nmax_tmp] = vtmp
            self.args.label_map_dict_rev_tot = {
                v: k for k, v in self.args.label_map_dict_tot.items()}
            # self.args.label_map_dict_rev_tot = {v: k for k,v in
            #   self.args.label_map_dict_tot.iteritems()}

            # infer lists from self.args
            self.args.yolt_object_labels2 = [
                self.args.label_map_dict2[ktmp]
                for ktmp in sorted(self.args.label_map_dict2.keys())]
            self.args.yolt_object_labels_str2 = ','.join(
                self.args.yolt_object_labels2)

            # set classnum and final output
            self.args.yolt_classnum2 = len(self.args.yolt_object_labels2)
            # for yolov2
            self.args.yolt_final_output2 = 1 * 1 * \
                self.args.boxes_per_grid * (self.args.yolt_classnum2 + 4 + 1)
            # for yolov3
            self.args.yolov3_filters2 = int(
                self.args.boxes_per_grid / 3 * (self.args.yolt_classnum2 + 4 + 1))

            # plot thresh and slice sizes
            self.args.plot_thresh2 = np.array(
                self.args.plot_thresh_str2.split(self.args.str_delim)).astype(float)
            self.args.slice_sizes2 = np.array(
                self.args.slice_sizes_str2.split(self.args.str_delim)).astype(int)

            # test files2
            self.args.yolt_test_classes_files2 = [
                os.path.join(self.args.results_dir, l + '.txt')
                for l in self.args.yolt_object_labels2]
            if len(self.args.test_presliced_list2) > 0:
                self.args.test_presliced_list_tot2 = os.path.join(
                    self.args.simrdwn_dir, self.args.test_presliced_list2)
            else:
                self.args.test_splitims_locs_file2 = os.path.join(
                    self.args.results_dir, self.args.test_splitims_locs_file_root2)
            self.args.test_tfrecord_out2 = os.path.join(
                self.args.results_dir, 'predictions2.tfrecord')
            self.args.val_df_path_init2 = os.path.join(
                self.args.results_dir, self.args.val_df_root_init2)
            self.args.val_df_path_aug2 = os.path.join(
                self.args.results_dir, self.args.val_df_root_aug2)
            self.args.weight_file_tot2 = os.path.join(
                self.args.results_topdir, self.args.train_model_path, self.args.weight_file2)
            self.args.yolt_cfg_file_tot2 = os.path.join(
                self.args.log_dir, self.args.yolt_cfg_file2)

            if self.args.mode == 'test':
                self.args.yolt_cfg_file_in2 = os.path.join(os.path.dirname(
                    self.args.weight_file_tot2), 'logs/', self.args.yolt_cfg_file2)
            else:
                self.args.yolt_cfg_file_in2 = os.path.join(
                    self.args.yolt_cfg_dir, self.args.yolt_cfg_file2)

            self.args.inference_graph_path_tot2 = os.path.join(
                self.args.results_topdir, self.args.train_model_path2,
                'frozen_model/frozen_inference_graph.pb')

        # total test
        self.args.val_df_path_tot = os.path.join(
            self.args.results_dir, self.args.val_df_root_tot)
        self.args.val_prediction_df_refine_tot = os.path.join(
            self.args.results_dir, self.args.val_prediction_df_refine_tot_root_part
            + '_thresh=' + str(self.args.plot_thresh[0]))

        # if evaluating spacenet
        if len(self.args.building_csv_file) > 0:
            self.args.building_csv_file = os.path.join(
                self.args.results_dir, self.args.building_csv_file)

        ##########################
        # Plotting params
        self.args.figsize = (12, 12)
        self.args.dpi = 300

        return self.args

    ###############################################################################

    def update_tf_train_config(self,
                               config_file_in, config_file_out,
                               label_map_path='',  train_tf_record='',
                               train_input_width=416, train_input_height=416,
                               train_val_tf_record='', num_steps=10000,
                               batch_size=32,
                               verbose=False):
        """
        Edit tf trainig config file to reflect proper paths and parameters

        Notes
        -----
        For details on how to set up the pipeline, see:
            https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md 
        For example .config files:
            https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
            also located at: /raid/cosmiq/simrdwn/configs

        Arguments
        ---------
        config_file_in : str
            Path to input config file.
        config_file_out : str
            Path to output config file.
        label_map_path : str
            Path to label map file (required).  Defaults to ``''`` (empty).
        ...

        """

        #############
        # for now, set train_val_tf_record to train_tf_record!
        train_val_tf_record = train_tf_record
        #############

        # load pbtxt
        label_map_dict = self.preprocess_tfrecords.load_pbtxt(
            label_map_path, verbose=False)
        n_classes = len(list(label_map_dict.keys()))

        # print ("config_file_in:", config_file_in)
        fin = open(config_file_in, 'r')
        fout = open(config_file_out, 'w')
        line_minus_two = ''
        line_list = []
        for i, line in enumerate(fin):
            if verbose:
                self.sig1.emit(str(i)+": " + line)
            line_list.append(line)

            # set line_minus_two
            if i > 1:
                line_minus_two = line_list[i-2].strip()

            # assume train_path is first, val_path is second
            if (line.strip().startswith('input_path:')) and (line_minus_two.startswith('train_input_reader:')):
                line_out = '    input_path: "' + str(train_tf_record) + '"\n'

            elif (line.strip().startswith('input_path:')) and (line_minus_two.startswith('eval_input_reader:')):
                line_out = '    input_path: "' + \
                    str(train_val_tf_record) + '"\n'

            elif line.strip().startswith('label_map_path:'):
                line_out = '  label_map_path: "' + str(label_map_path) + '"\n'

            elif line.strip().startswith('batch_size:'):
                line_out = '  batch_size: ' + str(batch_size) + '\n'

            elif line.strip().startswith('num_steps:'):
                line_out = '  num_steps: ' + str(num_steps) + '\n'
            # resizer
            elif line.strip().startswith('height:'):
                line_out = '        height: ' + str(train_input_height) + '\n'
            elif line.strip().startswith('width:'):
                line_out = '        width: ' + str(train_input_width) + '\n'
            # n classes
            elif line.strip().startswith('num_classes:'):
                line_out = '    num_classes: ' + str(n_classes) + '\n'

            else:
                line_out = line
            fout.write(line_out)

        fin.close()
        fout.close()

    ###############################################################################

    def tf_train_cmd(self, tf_cfg_train_file, results_dir, max_batches=10000):
        """
        Train a model with tensorflow object detection api
        Example:
        python /opt/tensorflow-models/research/object_detection/train.py \
            --logtostderr \
            --pipeline_config_path=tf/configs/ssd_inception_v2_simrdwn.config \
            --train_dir=outputs/ssd >> \
                train_ssd_inception_v2_simrdwn.log & tail -f train_ssd_inception_v2_simrdwn.log
        """

        # suffix = ' >> ' + log_file + ' & tail -f ' + log_file
        # suffix =  >> ' + log_file
        suffix = ''

        # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md
        # PIPELINE_CONFIG_PATH={path to pipeline config file}
        # MODEL_DIR={path to model directory}
        # NUM_TRAIN_STEPS=50000
        # SAMPLE_1_OF_N_EVAL_EXAMPLES=1
        # python object_detection/model_main.py \
        #    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
        #    --model_dir=${MODEL_DIR} \
        #    --num_train_steps=${NUM_TRAIN_STEPS} \
        #    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
        #    --alsologtostderr
        cmd_arg_list = [
            'python',
            '/opt/tensorflow-models/research/object_detection/model_main.py',
            '--pipeline_config_path=' + tf_cfg_train_file,
            '--model_dir=' + results_dir,
            '--num_train_steps=' + str(int(max_batches)),
            '--sample_1_of_n_eval_examples={}'.format(1),
            '--alsologtostderr',
            suffix
        ]

        # old version of tensorflow
        # cmd_arg_list = [
        #        'python',
        #        '/opt/tensorflow-models/research/object_detection/train.py',
        #        '--logtostderr',
        #        '--pipeline_config_path=' + tf_cfg_train_file,
        #        '--train_dir=' + results_dir,
        #        suffix
        #        ]

        cmd = ' '.join(cmd_arg_list)

        return cmd

    ###############################################################################
    def tf_export_model_cmd(self,
                            trained_dir='', tf_cfg_train_file='pipeline.config',
                            model_output_root='frozen_model', verbose=False):
        """Export trained model with tensorflow object detection api"""

        # get max training batches completed
        checkpoints_tmp = [ftmp for ftmp in os.listdir(trained_dir)
                           if ftmp.startswith('model.ckpt')]
        # print ("checkpoints tmp:", checkpoints_tmp)
        nums_tmp = [int(z.split('model.ckpt-')[-1].split('.')[0])
                    for z in checkpoints_tmp]
        # print ("nums_tmp:", nums_tmp)
        num_max_tmp = np.max(nums_tmp)
        if verbose:
            self.sig1.emit(
                "tf_export_model_cmd() - checkpoints_tmp: " + checkpoints_tmp)
            self.sig1.emit(
                "tf_export_model_cmd() - num_max_tmp: " + num_max_tmp)

        cmd_arg_list = [
            'python',
            '/opt/tensorflow-models/research/object_detection/export_inference_graph.py',
            # '/opt/tensorflow-models/research/object_detection/export_inference_graph.py',
            '--input_type image_tensor',
            '--pipeline_config_path=' + \
            os.path.join(trained_dir, tf_cfg_train_file),
            '--trained_checkpoint_prefix=' + \
            os.path.join(trained_dir, 'model.ckpt-' + str(num_max_tmp)),
            # '--trained_checkpoint_prefix=' + os.path.join(results_dir, 'model.ckpt-' + str(num_steps)),
            '--output_directory=' + \
            os.path.join(trained_dir, model_output_root)
        ]

        cmd = ' '.join(cmd_arg_list)
        if verbose:
            self.sig1.emit("tf_export_model_cmd() - output cmd: " + cmd)

        return cmd

    ###############################################################################

    def tf_infer_cmd_dual(self,
                          inference_graph_path='',
                          input_file_list='',
                          in_tfrecord_path='',
                          out_tfrecord_path='',
                          use_tfrecords=0,
                          min_thresh=0.05,
                          GPU=0,
                          BGR2RGB=0,
                          output_csv_path='',
                          infer_src_path='simrdwn/core'):
        """
        Run infer_detections.py with the given input tfrecord or input_file_list

        Infer output tfrecord
        Example:
            python simrdwn/src/infer_detections.py \
                    --input_tfrecord_paths=data/qgis_labels_car_boat_plane_val.tfrecord \
                    --inference_graph=outputs/ssd/output_inference_graph/frozen_inference_graph.pb \
                    --output_tfrecord_path=outputs/ssd/val_detections_ssd.tfrecord 
        df.to_csv(outfile_df)
        """

        cmd_arg_list = [
            'python',
            infer_src_path + '/' + 'infer_detections.py',
            '--inference_graph=' + inference_graph_path,
            '--GPU=' + str(GPU)
        ]
        if bool(use_tfrecords):
            cmd_arg_list.extend(['--use_tfrecord=' + str(use_tfrecords)])

        cmd_arg_list.extend([
            # first method, with tfrecords
            '--input_tfrecord_paths=' + in_tfrecord_path,
            '--output_tfrecord_path=' + out_tfrecord_path,
            # second method, with file list
            '--input_file_list=' + input_file_list,
            '--BGR2RGB=' + str(BGR2RGB),
            '--output_csv_path=' + output_csv_path,
            '--min_thresh=' + str(min_thresh)
        ])
        cmd = ' '.join(cmd_arg_list)

        return cmd

    ###############################################################################
    def yolt_command(self,
                     framework='yolt2',
                     yolt_cfg_file_tot='',
                     weight_file_tot='',
                     results_dir='',
                     log_file='',
                     yolt_loss_file='',
                     mode='train',
                     yolt_object_labels_str='',
                     yolt_classnum=1,
                     nbands=3,
                     gpu=0,
                     single_gpu_machine=0,
                     yolt_train_images_list_file_tot='',
                     test_splitims_locs_file='',
                     test_im_tot='',
                     test_thresh=0.2,
                     yolt_nms_thresh=0,
                     min_retain_prob=0.025):
        """
        Define YOLT commands
        yolt.c expects the following inputs:
        // arg 0 = GPU number
        // arg 1 'yolt'
        // arg 2 = mode
        char *cfg = argv[3];
        char *weights = (argc > 4) ? argv[4] : 0;
        char *test_filename = (argc > 5) ? argv[5]: 0;
        float plot_thresh = (argc > 6) ? atof(argv[6]): 0.2;
        float nms_thresh = (argc > 7) ? atof(argv[7]): 0;
        char *train_images = (argc > 8) ? argv[8]: 0;
        char *results_dir = (argc > 9) ? argv[9]: 0;
        //char *test_image = (argc >10) ? argv[10]: 0;
        char *test_list_loc = (argc > 10) ? argv[10]: 0;
        char *names_str = (argc > 11) ? argv[11]: 0;
        int len_names = (argc > 12) ? atoi(argv[12]): 0;
        int nbands = (argc > 13) ? atoi(argv[13]): 0;
        char *loss_file = (argc > 14) ? argv[14]: 0;
        """

        ##########################
        # set gpu command
        if single_gpu_machine == 1:  # use_aware:
            gpu_cmd = ''
        else:
            gpu_cmd = '-i ' + str(gpu)
            # gpu_cmd = '-i ' + str(3-self.args.gpu) # originally, numbers were reversed

        ##########################
        # SET VARIABLES ACCORDING TO MODE (SET UNNECCESSARY VALUES TO 0 OR NULL)
        # set train prams (and prefix, and suffix)
        if mode == 'train':
            mode_str = 'train'
            train_ims = yolt_train_images_list_file_tot
            prefix = 'nohup'
            suffix = ' >> ' + log_file + ' & tail -f ' + log_file
        else:
            train_ims = 'null'
            prefix = ''
            suffix = ' 2>&1 | tee -a ' + log_file

        # set test deprecated params
        if mode == 'test_deprecated':
            test_im = test_im_tot
            test_thresh = test_thresh
        else:
            test_im = 'null'
            test_thresh = 0

        # set test params
        if mode == 'test':
            mode_str = 'valid'
            # test_image = self.args.test_image_tmp
            test_list_loc = test_splitims_locs_file
        else:
            # test_image = 'null'
            test_list_loc = 'null'

        ##########################

        c_arg_list = [
            prefix,
            './' + framework.lower() + '/darknet',
            gpu_cmd,
            framework,  # 'yolt2',
            mode_str,
            yolt_cfg_file_tot,
            weight_file_tot,
            test_im,
            str(test_thresh),
            str(yolt_nms_thresh),
            train_ims,
            results_dir,
            test_list_loc,
            yolt_object_labels_str,
            str(yolt_classnum),
            str(nbands),
            yolt_loss_file,
            str(min_retain_prob),
            suffix
        ]

        cmd = ' '.join(c_arg_list)

        self.sig1.emit("Command:\n" + cmd)

        return cmd

    ###############################################################################

    def recompile_darknet(self, yolt_dir):
        """compile darknet"""
        os.chdir(yolt_dir)
        cmd_compile0 = 'make clean'
        cmd_compile1 = 'make'

        self.sig1.emit(cmd_compile0)
        utils._run_cmd(cmd_compile0)

        self.sig1.emit(cmd_compile1)
        utils._run_cmd(cmd_compile1)

    ###############################################################################

    def replace_yolt_vals_train_compile(self, framework='yolt2',
                                        yolt_dir='',
                                        mode='train',
                                        yolt_cfg_file_tot='',
                                        yolt_final_output='',
                                        yolt_classnum=2,
                                        nbands=3,
                                        max_batches=512,
                                        batch_size=16,
                                        subdivisions=4,
                                        boxes_per_grid=5,
                                        train_input_width=416,
                                        train_input_height=416,
                                        yolov3_filters=0,
                                        use_GPU=1,
                                        use_opencv=1,
                                        use_CUDNN=1):
        """
        For either training or compiling,
        edit cfg file in darknet to allow for custom models
        editing of network layers must be done in vi, this function just changes
        parameters such as window size, number of trianing steps, etc
        """

        self.sig1.emit("Replacing YOLT vals...")

        #################
        # Makefile
        if mode == 'compile':
            yoltm = os.path.join(yolt_dir, 'Makefile')
            yoltm_tmp = yoltm + 'tmp'
            f1 = open(yoltm, 'r')
            f2 = open(yoltm_tmp, 'w')
            for line in f1:
                if line.strip().startswith('GPU='):
                    line_out = 'GPU=' + str(use_GPU) + '\n'
                elif line.strip().startswith('OPENCV='):
                    line_out = 'OPENCV=' + str(use_opencv) + '\n'
                elif line.strip().startswith('CUDNN='):
                    line_out = 'CUDNN=' + str(use_CUDNN) + '\n'
                else:
                    line_out = line
                f2.write(line_out)
            f1.close()
            f2.close()
            # copy old yoltm
            utils._run_cmd('cp ' + yoltm + ' ' + yoltm + '_v0')
            # write new file over old
            utils._run_cmd('mv ' + yoltm_tmp + ' ' + yoltm)

        #################
        # cfg file
        elif mode == 'train':
            yoltcfg = yolt_cfg_file_tot
            # print ("\n\nyolt_cfg_file_tot:", yolt_cfg_file_tot)
            yoltcfg_tmp = yoltcfg + 'tmp'
            f1 = open(yoltcfg, 'r')
            f2 = open(yoltcfg_tmp, 'w')
            # read in reverse because we want to edit the last output length
            s = f1.readlines()
            s.reverse()
            sout = []

            fixed_output = False
            for i, line in enumerate(s):
                # print ("line:", line)

                if i > 3:
                    lm4 = sout[i-4]
                else:
                    lm4 = ''

                # if line.strip().startswith('side='):
                #    line_out='side=' + str(side) + '\n'
                if line.strip().startswith('channels='):
                    line_out = 'channels=' + str(nbands) + '\n'
                elif line.strip().startswith('classes='):
                    line_out = 'classes=' + str(yolt_classnum) + '\n'
                elif line.strip().startswith('max_batches'):
                    line_out = 'max_batches=' + str(max_batches) + '\n'
                elif line.strip().startswith('batch='):
                    line_out = 'batch=' + str(batch_size) + '\n'
                elif line.strip().startswith('subdivisions='):
                    line_out = 'subdivisions=' + str(subdivisions) + '\n'

                # replace num in yolov2
                elif (framework.upper() == 'YOLT2') and (line.strip().startswith('num=')):
                    line_out = 'num=' + str(boxes_per_grid) + '\n'
                elif (framework.upper() in ['YOLT2', 'YOLT3']) and line.strip().startswith('width='):
                    line_out = 'width=' + str(train_input_width) + '\n'
                elif (framework.upper() in ['YOLT2', 'YOLT3']) and line.strip().startswith('height='):
                    line_out = 'height=' + str(train_input_height) + '\n'
                # change final output, and set fixed to true
                # elif (line.strip().startswith('output=')) and (not fixed_output):
                #    line_out = 'output=' + str(final_output) + '\n'
                #    fixed_output=True
                elif (framework.upper() == 'YOLT2') and (line.strip().startswith('filters=')) and (not fixed_output):
                    line_out = 'filters=' + str(yolt_final_output) + '\n'
                    fixed_output = True

                # line before a yolo layer should have 3 * (n_classes + 5) filters
                elif (framework.upper() == 'YOLT3') and (line.strip().startswith('filters=')) and (lm4.startswith('[yolo]')):
                    line_out = 'filters=' + str(yolov3_filters) + '\n'
                    self.sig1.emit(str(i) + " lm4: " + lm4 +
                                   " line_out: " + line_out)
                    self.sig1.emit(
                        "self.args.yolov3_filters: " + str(yolov3_filters))
                    # return

                else:
                    line_out = line
                sout.append(line_out)

            sout.reverse()
            for line in sout:
                f2.write(line)

            f1.close()
            f2.close()

            # copy old yoltcfg?
            utils._run_cmd('cp ' + yoltcfg + ' ' + yoltcfg[:-4] + 'orig.cfg')
            # write new file over old
            utils._run_cmd('mv ' + yoltcfg_tmp + ' ' + yoltcfg)
        #################

        else:
            return

    ###############################################################################

    def split_test_im(self,
                      im_root_with_ext, testims_dir_tot, results_dir,
                      log_file,
                      slice_sizes=[416],
                      slice_overlap=0.2,
                      test_slice_sep='__',
                      zero_frac_thresh=0.5,
                      ):
        """
        Split files for test step
        Assume input string has no path, but does have extension (e.g:, 'pic.png')

        1. get image path (self.args.test_image_tmp) from image root name
                (self.args.test_image_tmp)
        2. slice test image and move to results dir
        """

        # get image root, make sure there is no extension
        im_root = im_root_with_ext.split('.')[0]
        im_path = os.path.join(testims_dir_tot, im_root_with_ext)

        # slice test plot into manageable chunks

        # slice (if needed)
        if slice_sizes[0] > 0:
            # if len(self.args.slice_sizes) > 0:
            # create test_splitims_locs_file
            # set test_dir as in results_dir
            test_split_dir = os.path.join(
                results_dir,  im_root + '_split' + '/')
            test_dir_str = '"test_split_dir: ' + test_split_dir + '\n"'
            self.sig1.emit("test_dir: " + test_dir_str[1:-2])
            os.system('echo ' + test_dir_str + ' >> ' + log_file)
            # print "test_split_dir:", test_split_dir

            # clean out dir, and make anew
            if os.path.exists(test_split_dir):
                if (not test_split_dir.startswith(results_dir)) \
                        or len(test_split_dir) < len(results_dir) \
                        or len(test_split_dir) < 10:
                    self.sig1.emit(
                        "test_split_dir too short!!!!:" + test_split_dir)
                    return
                shutil.rmtree(test_split_dir, ignore_errors=True)
            os.mkdir(test_split_dir)

            # slice
            for s in slice_sizes:
                self.slice_im(im_path, im_root,
                              test_split_dir, s, s,
                              zero_frac_thresh=zero_frac_thresh,
                              overlap=slice_overlap,
                              slice_sep=test_slice_sep)
                test_files = [os.path.join(test_split_dir, f) for
                              f in os.listdir(test_split_dir)]
            n_files_str = '"Num files: ' + str(len(test_files)) + '\n"'
            self.sig1.emit(n_files_str[1:-2])
            os.system('echo ' + n_files_str + ' >> ' + log_file)

        else:
            test_files = [im_path]
            test_split_dir = os.path.join(results_dir, 'nonsense')

        return test_files, test_split_dir

    ###############################################################################

    def prep_test_files(self, results_dir, log_file, test_ims_list,
                        testims_dir_tot, test_splitims_locs_file,
                        slice_sizes=[416],
                        slice_overlap=0.2,
                        test_slice_sep='__',
                        zero_frac_thresh=0.5,
                        ):
        """Split images and save split image locations to txt file"""

        # split test images, store locations
        t0 = time.time()
        test_split_str = '"Splitting test files...\n"'
        self.sig1.emit(test_split_str[1:-2])
        os.system('echo ' + test_split_str + ' >> ' + log_file)
        self.sig1.emit("test_ims_list:" + str(test_ims_list))

        test_files_locs_list = []
        test_split_dir_list = []
        # !! Should make a tfrecord when we split files, instead of doing it later
        for i, test_base_tmp in enumerate(test_ims_list):
            iter_string = '"\n' + str(i+1) + ' / ' + \
                str(len(test_ims_list)) + '\n"'
            self.sig1.emit(iter_string[1:-2])
            os.system('echo ' + iter_string + ' >> ' + log_file)
            # print "\n", i+1, "/", len(self.args.test_ims_list)

            # dirty hack: ignore file extensions for now
            # test_base_tmp_noext = test_base_tmp.split('.')[0]
            # test_base_string = '"test_base_tmp_noext:' \
            #                    + str(test_base_tmp_noext) + '\n"'
            test_base_string = '"test_file: ' + str(test_base_tmp) + '\n"'
            self.sig1.emit(test_base_string[1:-2])
            os.system('echo ' + test_base_string + ' >> ' + log_file)

            # split data
            # test_files_list_tmp, test_split_dir_tmp = split_test_im(test_base_tmp, self.args)
            test_files_list_tmp, test_split_dir_tmp = \
                self.split_test_im(test_base_tmp, testims_dir_tot,
                                   results_dir, log_file,
                                   slice_sizes=slice_sizes,
                                   slice_overlap=slice_overlap,
                                   test_slice_sep=test_slice_sep,
                                   zero_frac_thresh=zero_frac_thresh)
            # add test_files to list
            test_files_locs_list.extend(test_files_list_tmp)
            test_split_dir_list.append(test_split_dir_tmp)

        # swrite test_files_locs_list to file (file = test_splitims_locs_file)
        self.sig1.emit("Total len test files:" +
                       str(len(test_files_locs_list)))
        self.sig1.emit("test_splitims_locs_file:" + test_splitims_locs_file)
        # write list of files to test_splitims_locs_file
        with open(test_splitims_locs_file, "w") as fp:
            for line in test_files_locs_list:
                if not line.endswith('.DS_Store'):
                    fp.write(line + "\n")

        t1 = time.time()
        cmd_time_str = '"\nLength of time to split test files: ' \
            + str(t1 - t0) + ' seconds\n"'
        self.sig1.emit(cmd_time_str[1:-2])
        os.system('echo ' + cmd_time_str + ' >> ' + log_file)

        return test_files_locs_list, test_split_dir_list

    ###############################################################################

    def run_test(self, framework='YOLT2',
                 infer_cmd='',
                 results_dir='',
                 log_file='',
                 n_files=0,
                 test_tfrecord_out='',
                 slice_sizes=[416],
                 testims_dir_tot='',
                 yolt_test_classes_files='',
                 label_map_dict={},
                 val_df_path_init='',
                 test_slice_sep='__',
                 edge_buffer_test=1,
                 max_edge_aspect_ratio=4,
                 test_box_rescale_frac=1.0,
                 rotate_boxes=False,
                 min_retain_prob=0.05,
                 test_add_geo_coords=True,
                 verbose=False
                 ):
        """Evaluate multiple large images"""

        t0 = time.time()
        # run command
        self.sig1.emit("Running " + infer_cmd)
        os.system('echo ' + infer_cmd + ' >> ' + log_file)
        os.system(infer_cmd)  # run_cmd(outcmd)
        t1 = time.time()
        cmd_time_str = '"\nLength of time to run command: ' + infer_cmd \
            + ' for ' + str(n_files) + ' cutouts: ' \
            + str(t1 - t0) + ' seconds\n"'
        self.sig1.emit(cmd_time_str[1:-1])
        os.system('echo ' + cmd_time_str + ' >> ' + log_file)

        if framework.upper() not in ['YOLT2', 'YOLT3']:

            # if we ran inference with a tfrecord, we must now parse that into
            #   a dataframe
            if len(test_tfrecord_out) > 0:
                df_init = parse_tfrecord.tf_to_df(
                    test_tfrecord_out, max_iter=500000,
                    label_map_dict=label_map_dict, tf_type='test',
                    output_columns=['Loc_Tmp', u'Prob', u'Xmin', u'Ymin',
                                    u'Xmax', u'Ymax', u'Category'],
                    # replace_paths=()
                )
                # use numeric categories
                label_map_dict_rev = {v: k for k, v in label_map_dict.items()}
                # label_map_dict_rev = {v: k for k,v in label_map_dict.iteritems()}
                df_init['Category'] = [label_map_dict_rev[vtmp]
                                       for vtmp in df_init['Category'].values]
                # save to file
                df_init.to_csv(val_df_path_init)
            else:
                self.sig1.emit("Read in val_df_path_init: " + val_df_path_init)
                df_init = pd.read_csv(val_df_path_init
                                      # names=[u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin',
                                      #       u'Xmax', u'Ymax', u'Category']
                                      )

            #########
            # post process
            self.sig1.emit("len df_init:" + str(len(df_init)))
            df_init.index = np.arange(len(df_init))

            # clean out low probabilities
            self.sig1.emit("minimum retained threshold:" +
                           str(min_retain_prob))
            bad_idxs = df_init[df_init['Prob'] < min_retain_prob].index
            if len(bad_idxs) > 0:
                self.sig1.emit("bad idxss:"+str(bad_idxs))
                df_init.drop(df_init.index[bad_idxs], inplace=True)

            # clean out bad categories
            df_init['Category'] = df_init['Category'].values.astype(int)
            good_cats = list(label_map_dict.keys())
            self.sig1.emit("Allowed categories:"+str(good_cats))
            # print ("df_init0['Category'] > np.max(good_cats)", df_init['Category'] > np.max(good_cats))
            # print ("df_init0[df_init0['Category'] > np.max(good_cats)]", df_init[df_init['Category'] > np.max(good_cats)])
            bad_idxs2 = df_init[df_init['Category'] > np.max(good_cats)].index
            if len(bad_idxs2) > 0:
                self.sig1.emit("label_map_dict:" + str(label_map_dict))
                self.sig1.emit("df_init['Category']:" +
                               str(df_init['Category']))
                self.sig1.emit("bad idxs2: " + str(bad_idxs2))
                df_init.drop(df_init.index[bad_idxs2], inplace=True)

            # set index as sequential
            df_init.index = np.arange(len(df_init))

            # df_init = df_init0[df_init0['Category'] <= np.max(good_cats)]
            # if (len(df_init) != len(df_init0)):
            #    print (len(df_init0) - len(df_init), "rows cleaned out")

            # tf_infer_cmd outputs integer categories, update to strings
            df_init['Category'] = [label_map_dict[ktmp]
                                   for ktmp in df_init['Category'].values]

            self.sig1.emit("len df_init after filtering: " + str(len(df_init)))

            # augment dataframe columns
            df_tot = post_process.augment_df(
                df_init,
                testims_dir_tot=self.testims_dir_tot,
                slice_sizes=slice_sizes,
                slice_sep=test_slice_sep,
                edge_buffer_test=edge_buffer_test,
                max_edge_aspect_ratio=max_edge_aspect_ratio,
                test_box_rescale_frac=test_box_rescale_frac,
                rotate_boxes=rotate_boxes,
                verbose=True)

        else:
            # post-process
            # df_tot = post_process_yolt_test_create_df(self.args)
            df_tot = post_process.post_process_yolt_test_create_df(
                yolt_test_classes_files, log_file,
                testims_dir_tot=testims_dir_tot,
                slice_sizes=slice_sizes,
                slice_sep=test_slice_sep,
                edge_buffer_test=edge_buffer_test,
                max_edge_aspect_ratio=max_edge_aspect_ratio,
                test_box_rescale_frac=test_box_rescale_frac,
                rotate_boxes=rotate_boxes)

        ###########################################
        # plot

        # add geo coords to eall boxes?
        if test_add_geo_coords and len(df_tot) > 0:
            ###########################################
            # !!!!! Skip?
            # json = None
            ###########################################
            df_tot, json = add_geo_coords.add_geo_coords_to_df(
                df_tot, create_geojson=False, inProj_str='epsg:4326',
                outProj_str='epsg:3857', verbose=verbose)
        else:
            json = None

        return df_tot, json

    ###############################################################################

    def prep(self):
        """Prep data for train or test

        Arguments
        ---------
        self.args : Namespace
            input arguments

        Returns
        -------
        train_cmd1 : str
            Training command
        test_cmd_tot : str
            Testing command
        test_cmd_tot2 : str
            Testing command for second scale (optional)
        """

        # initialize commands to null strings
        train_cmd1, test_cmd_tot, test_cmd_tot2 = '', '', ''

        self.sig1.emit("\nSIMRDWN now...\n")
        os.chdir(self.args.simrdwn_dir)
        self.sig1.emit("cwd: " + os.getcwd())
        # t0 = time.time()

        # make dirs
        os.mkdir(self.args.results_dir)
        os.mkdir(self.args.log_dir)

        # create log file
        self.sig1.emit("Date string: " + self.args.date_string)
        os.system('echo ' + str(self.args.date_string) +
                  ' > ' + self.args.log_file)
        # init to the contents in this file?
        # os.system('cat ' + self.args.this_file + ' >> ' + self.args.log_file)
        args_str = '"\nArgs: ' + str(self.args) + '\n"'
        # self.sig1.emit(args_str[1:-1])
        os.system('echo ' + args_str + ' >> ' + self.args.log_file)

        # copy this file (yolt_run.py) as well as config, plot file to results_dir
        shutil.copy2(self.args.this_file, self.args.log_dir)
        # shutil.copy2(self.args.yolt_plot_file, self.args.log_dir)
        # shutil.copy2(self.args.tf_plot_file, self.args.log_dir)
        self.sig1.emit("log_dir: " + self.args.log_dir)

        # print ("\nlabel_map_dict:", self.args.label_map_dict)
        self.sig1.emit("\nlabel_map_dict_tot: " +
                       str(self.args.label_map_dict_tot))
        # print ("object_labels:", self.args.object_labels)
        self.sig1.emit("yolt_object_labels: " +
                       str(self.args.yolt_object_labels))
        self.sig1.emit("yolt_classnum: "+str(self.args.yolt_classnum))

        # save labels to log_dir
        # pickle.dump(self.args.object_labels, open(self.args.log_dir \
        #                                + 'labels_list.pkl', 'wb'), protocol=2)
        with open(self.args.labels_log_file, "w") as fp:
            for ob in self.args.yolt_object_labels:
                fp.write(str(ob) + "\n")

        # set YOLT values, if desired
        if (self.args.framework.upper() == 'YOLT2') \
                or (self.args.framework.upper() == 'YOLT3'):

            # copy files to log dir
            shutil.copy2(self.args.yolt_plot_file, self.args.log_dir)
            shutil.copy2(self.args.yolt_cfg_file_in, self.args.log_dir)
            os.system('cat ' + self.args.yolt_cfg_file_tot +
                      ' >> ' + self.args.log_file)
            # print config values
            # self.sig1.emit("yolt_cfg_file: "+ self.args.yolt_cfg_file_in)
            if self.args.mode.upper() in ['TRAIN', 'COMPILE']:
                self.sig1.emit("Updating yolt params in files...")
                self.replace_yolt_vals_train_compile(
                    framework=self.args.framework,
                    yolt_dir=self.args.yolt_dir,
                    mode=self.args.mode,
                    yolt_cfg_file_tot=self.args.yolt_cfg_file_tot,
                    yolt_final_output=self.args.yolt_final_output,
                    yolt_classnum=self.args.yolt_classnum,
                    nbands=self.args.nbands,
                    yolov3_filters=self.args.yolov3_filters,
                    max_batches=self.args.max_batches,
                    batch_size=self.args.batch_size,
                    subdivisions=self.args.subdivisions,
                    boxes_per_grid=self.args.boxes_per_grid,
                    train_input_width=self.args.train_input_width,
                    train_input_height=self.args.train_input_height,
                    use_GPU=self.args.use_GPU,
                    use_opencv=self.args.use_opencv,
                    use_CUDNN=self.args.use_CUDNN)
                # replace_yolt_vals(self.args)
                # print a few values...
                self.sig1.emit("Final output layer size: " +
                               str(self.args.yolt_final_output))
                # print ("side size:", self.args.side)
                self.sig1.emit("batch_size: " + str(self.args.batch_size))
                self.sig1.emit("subdivisions: " + str(self.args.subdivisions))

            if self.args.mode.upper() == 'COMPILE':
                self.sig1.emit("Recompiling yolt...")
                recompile_darknet(self.args.yolt_dir)
                return

            # set yolt command
            yolt_cmd = self.yolt_command(
                self.args.framework, yolt_cfg_file_tot=self.args.yolt_cfg_file_tot,
                weight_file_tot=self.args.weight_file_tot,
                results_dir=self.args.results_dir,
                log_file=self.args.log_file,
                yolt_loss_file=self.args.yolt_loss_file,
                mode=self.args.mode,
                yolt_object_labels_str=self.args.yolt_object_labels_str,
                yolt_classnum=self.args.yolt_classnum,
                nbands=self.args.nbands,
                gpu=self.args.gpu,
                single_gpu_machine=self.args.single_gpu_machine,
                yolt_train_images_list_file_tot=self.args.yolt_train_images_list_file_tot,
                test_splitims_locs_file=self.args.test_splitims_locs_file,
                yolt_nms_thresh=self.args.yolt_nms_thresh,
                min_retain_prob=self.args.min_retain_prob)

            if self.args.mode.upper() == 'TRAIN':
                self.sig1.emit("yolt_train_cmd: " + yolt_cmd)
                # train_cmd_tot = yolt_cmd
                train_cmd1 = yolt_cmd
                # train_cmd2 = ''
            # set second test command
            elif self.args.mode.upper() == 'TEST':
                test_cmd_tot = yolt_cmd
            else:
                self.sig1.emit(
                    "Error: Unknown execution type (should be train or test)")
                return

            if len(self.args.label_map_path2) > 0:
                test_cmd_tot2 = self.yolt_command(
                    self.args.framework, yolt_cfg_file_tot=self.args.yolt_cfg_file_tot2,
                    weight_file_tot=self.args.weight_file_tot2,
                    results_dir=self.args.results_dir,
                    log_file=self.args.log_file,
                    mode=self.args.mode,
                    yolt_object_labels_str=self.args.yolt_object_labels_str2,
                    classnum=self.args.yolt_classnum2,
                    nbands=self.args.nbands,
                    gpu=self.args.gpu,
                    single_gpu_machine=self.args.single_gpu_machine,
                    test_splitims_locs_file=self.args.test_splitims_locs_file2,
                    yolt_nms_thresh=self.args.yolt_nms_thresh,
                    min_retain_prob=self.args.min_retain_prob)

            else:
                test_cmd_tot2 = ''

        # set tensor flow object detection API values
        else:

            if self.args.mode.upper() == 'TRAIN':
                if not os.path.exists(self.args.tf_model_output_directory):
                    os.mkdir(self.args.tf_model_output_directory)
                # copy plot file to output dir
                shutil.copy2(self.args.tf_plot_file, self.args.log_dir)

                self.sig1.emit("Updating tf_config...")
                self.update_tf_train_config(
                    self.args.tf_cfg_train_file, self.args.tf_cfg_train_file_out,
                    label_map_path=self.args.label_map_path,
                    train_tf_record=self.args.train_tf_record,
                    train_val_tf_record=self.args.train_val_tf_record,
                    train_input_width=self.args.train_input_width,
                    train_input_height=self.args.train_input_height,
                    batch_size=self.args.batch_size,
                    num_steps=self.args.max_batches)
                # define train command
                cmd_train_tf = self.tf_train_cmd(self.args.tf_cfg_train_file_out,
                                                 self.args.results_dir,
                                                 self.args.max_batches)

                # export command
                # cmd_export_tf = ''
                # cmd_export_tf = tf_export_model_cmd(self.args.tf_cfg_train_file_out,
                #                                 self.args.results_dir,
                #                                 self.args.tf_model_output_directory)
                #                                 #num_steps=self.args.max_batches)

                # # https://unix.stackexchange.com/questions/47230/how-to-execute-multiple-command-using-nohup/47249
                # # tot_cmd = nohup sh -c './cmd2 >result2 && ./cmd1 >result1' &
                # train_cmd_tot = 'nohup sh -c ' + "'" \
                #                    + cmd_train_tf + ' >> ' + self.args.log_file \
                #                    + ' && ' \
                #                    + cmd_export_tf + ' >> ' + self.args.log_file \
                #                    + "'" + ' & tail -f ' + self.args.log_file

                # train_cmd1 = cmd_train_tf
                train_cmd1 = 'nohup ' + cmd_train_tf + ' >> ' + self.args.log_file \
                    + ' & tail -f ' + self.args.log_file + ' &'

                # train_cmd2 = 'nohup ' +  cmd_export_tf + ' >> ' + self.args.log_file \
                #                + ' & tail -f ' + self.args.log_file #+ ' &'

                # forget about nohup since we're inside docker?
                # train_cmd1 = cmd_train_tf
                # train_cmd2 = cmd_export_tf

            # test
            else:

                # define inference (test) command   (output to csv)
                test_cmd_tot = self.tf_infer_cmd_dual(
                    inference_graph_path=self.args.inference_graph_path_tot,
                    input_file_list=self.args.test_splitims_locs_file,
                    in_tfrecord_path=self.args.test_presliced_tfrecord_tot,
                    out_tfrecord_path=self.args.test_tfrecord_out,
                    output_csv_path=self.args.val_df_path_init,
                    min_thresh=self.args.min_retain_prob,
                    BGR2RGB=self.args.BGR2RGB,
                    use_tfrecords=self.args.use_tfrecords,
                    infer_src_path=self.args.core_dir)

                # if using dual classifiers
                if len(self.args.label_map_path2) > 0:
                    # check if model exists, if not, create it.
                    if not os.path.exists(self.args.inference_graph_path_tot2):
                        inference_graph_path_tmp = os.path.dirname(
                            self.args.inference_graph_path_tot2)
                        cmd_tmp = 'python  ' \
                            + self.args.core_dir + '/export_model.py ' \
                            + '--results_dir ' + inference_graph_path_tmp
                        t1 = time.time()
                        self.sig1.emit("Running " + cmd_tmp + "...\n")
                        os.system(cmd_tmp)
                        t2 = time.time()
                        cmd_time_str = '"Length of time to run command: ' \
                            + cmd_tmp + ' ' \
                            + str(t2 - t1) + ' seconds\n"'
                        self.sig1.emit(cmd_time_str[1:-1])
                        os.system('echo ' + cmd_time_str +
                                  ' >> ' + self.args.log_file)
                    # set inference command
                    test_cmd_tot2 = self.tf_infer_cmd_dual(
                        inference_graph_path=self.args.inference_graph_path_tot2,
                        input_file_list=self.args.test_splitims_locs_file2,
                        output_csv_path=self.args.val_df_path_init2,
                        min_thresh=self.args.min_retain_prob,
                        GPU=self.args.gpu,
                        BGR2RGB=self.args.BGR2RGB,
                        use_tfrecords=self.args.use_tfrecords,
                        infer_src_path=self.args.core_dir)
                else:
                    test_cmd_tot2 = ''

        return train_cmd1, test_cmd_tot, test_cmd_tot2

    ###############################################################################

    def execute(self, train_cmd1, test_cmd_tot, test_cmd_tot2=''):
        """
        Execute train or test

        Arguments
        ---------
        train_cmd1 : str
            Training command
        test_cmd_tot : str
            Testing command
        test_cmd_tot2 : str
            Testing command for second scale (optional)

        Returns
        -------
        None
        """

        # Execute
        if self.args.mode.upper() == 'TRAIN':

            t1 = time.time()
            self.sig1.emit("Running " + train_cmd1 + "...\n\n")

            os.system(train_cmd1)
            # utils._run_cmd(train_cmd1)
            t2 = time.time()
            cmd_time_str = '"Length of time to run command: ' \
                + train_cmd1 + ' ' \
                + str(t2 - t1) + ' seconds\n"'
            self.sig1.emit(cmd_time_str[1:-1])
            os.system('echo ' + cmd_time_str + ' >> ' + self.args.log_file)

            # export trained model, if using tf object detection api?
            if 2 < 1 and (self.args.framework.upper() not in ['YOLT2', 'YOLT3']):
                cmd_export_tf = self.tf_export_model_cmd(
                    self.args.tf_cfg_train_file_out,
                    tf_cfg_train_file=self.args.tf_cfg_train_file,
                    model_output_root='frozen_model')
                # self.args.results_dir,
                # self.args.tf_model_output_directory,
                # tf_cfg_train_file=self.args.tf_cfg_train_file)
                train_cmd2 = cmd_export_tf

                t1 = time.time()
                self.sig1.emit("Running" + train_cmd2 + "...\n\n")
                # utils._run_cmd(train_cmd2)
                os.system(train_cmd2)
                t2 = time.time()
                cmd_time_str = '"Length of time to run command: ' \
                    + train_cmd2 + ' ' \
                    + str(t2 - t1) + ' seconds\n"'
                self.sig1.emit(cmd_time_str[1:-1])
                os.system('echo ' + cmd_time_str + ' >> ' + self.args.log_file)

        # need to split file for test first, then run command
        elif self.args.mode.upper() == 'TEST':

            t3 = time.time()
            # load presliced data, if desired
            if len(self.args.test_presliced_list) > 0:
                self.sig1.emit("Loading self.args.test_presliced_list: " +
                               str(self.args.test_presliced_list_tot))
                ftmp = open(self.args.test_presliced_list_tot, 'r')
                test_files_locs_list = [line.strip()
                                        for line in ftmp.readlines()]
                ftmp.close()
                test_split_dir_list = []
                self.sig1.emit("len test_files_locs_list: " +
                               str(len(test_files_locs_list)))
            elif len(self.args.test_presliced_tfrecord_path) > 0:
                self.sig1.emit(
                    "Using " + self.args.test_presliced_tfrecord_path)
                test_split_dir_list = []
            # split large test files
            else:
                self.sig1.emit("Prepping test files")
                test_files_locs_list, test_split_dir_list =\
                    self.prep_test_files(self.args.results_dir, self.args.log_file,
                                         self.args.test_ims_list,
                                         self.args.testims_dir_tot,
                                         self.args.test_splitims_locs_file,
                                         slice_sizes=self.args.slice_sizes,
                                         slice_overlap=self.args.slice_overlap,
                                         test_slice_sep=self.args.test_slice_sep,
                                         zero_frac_thresh=self.args.zero_frac_thresh,
                                         )
                # return if only interested in prepping
                if (bool(self.args.test_prep_only)) and (bool(self.args.use_tfrecords)):
                        # or (self.args.framework.upper() not in ['YOLT2', 'YOLT3']):
                    self.sig1.emit("Convert to tfrecords...")
                    TF_RecordPath = os.path.join(
                        self.args.results_dir, 'test_splitims.tfrecord')
                    self.preprocess_tfrecords.yolt_imlist_to_tf(
                        self.args.test_splitims_locs_file,
                        self.args.label_map_dict, TF_RecordPath,
                        TF_PathVal='', val_frac=0.0,
                        convert_dict={}, verbose=False)
                    self.sig1.emit("Done prepping test files, ending")
                    return

            # check if trained model exists, if not, create it.
            if (self.args.framework.upper() not in ['YOLT2', 'YOLT3']) and (not (os.path.exists(self.args.inference_graph_path_tot)) or
                                                                            (self.args.overwrite_inference_graph != 0)):
                self.sig1.emit("Creating self.args.inference_graph_path_tot: " +
                               self.args.inference_graph_path_tot, "...")

                # remove "saved_model" directory
                saved_dir = os.path.join(
                    os.path.dirname(self.args.inference_graph_path_tot), 'saved_model')
                self.sig1.emit("Removing " + saved_dir +
                               "so we can overwrite it...")
                if os.path.exists(saved_dir):
                    shutil.rmtree(saved_dir, ignore_errors=True)

                trained_dir_tmp = os.path.dirname(
                    os.path.dirname(self.args.inference_graph_path_tot))
                cmd_tmp = self.tf_export_model_cmd(
                    trained_dir=trained_dir_tmp,
                    tf_cfg_train_file=self.args.tf_cfg_train_file)

                # cmd_tmp = 'python  ' \
                #            + self.args.core_dir + '/export_model.py ' \
                #            + '--results_dir=' + inference_graph_path_tmp
                t1 = time.time()
                self.sig1.emit("Running export command: " + cmd_tmp + "...\n")
                os.system(cmd_tmp)
                t2 = time.time()
                cmd_time_str = '"Length of time to run command: ' \
                    + cmd_tmp + ' ' \
                    + str(t2 - t1) + ' seconds\n"'
                self.sig1.emit(cmd_time_str[1:-1])
                os.system('echo ' + cmd_time_str + ' >> ' + self.args.log_file)

            df_tot, json = self.run_test(infer_cmd=test_cmd_tot,
                                         framework=self.args.framework,
                                         results_dir=self.args.results_dir,
                                         log_file=self.args.log_file,
                                         # test_files_locs_list=test_files_locs_list,
                                         # test_presliced_tfrecord_tot=self.args.test_presliced_tfrecord_tot,
                                         test_tfrecord_out=self.args.test_tfrecord_out,
                                         slice_sizes=self.args.slice_sizes,
                                         testims_dir_tot=self.args.testims_dir_tot,
                                         yolt_test_classes_files=self.args.yolt_test_classes_files,
                                         label_map_dict=self.args.label_map_dict,
                                         val_df_path_init=self.args.val_df_path_init,
                                         # val_df_path_aug=self.args.val_df_path_aug,
                                         min_retain_prob=self.args.min_retain_prob,
                                         test_slice_sep=self.args.test_slice_sep,
                                         edge_buffer_test=self.args.edge_buffer_test,
                                         max_edge_aspect_ratio=self.args.max_edge_aspect_ratio,
                                         test_box_rescale_frac=self.args.test_box_rescale_frac,
                                         rotate_boxes=self.args.rotate_boxes,
                                         test_add_geo_coords=self.args.test_add_geo_coords)

            if len(df_tot) == 0:
                self.sig1.emit("No detections found!")
            else:
                # save to csv
                df_tot.to_csv(self.args.val_df_path_aug, index=False)
                # get number of files
                n_files = len(np.unique(df_tot['Loc_Tmp'].values))
                # n_files = str(len(test_files_locs_list)
                t4 = time.time()
                cmd_time_str = '"Length of time to run test for ' \
                    + str(n_files) + ' files = ' \
                    + str(t4 - t3) + ' seconds\n"'
                self.sig1.emit(cmd_time_str[1:-1])
                os.system('echo ' + cmd_time_str + ' >> ' + self.args.log_file)

            # run again, if desired
            if len(self.args.weight_file2) > 0:

                t5 = time.time()
                # split large testion files
                self.sig1.emit("Prepping test files")
                test_files_locs_list2, test_split_dir_list2 =\
                    self.prep_test_files(self.args.results_dir, self.args.log_file,
                                         self.args.test_ims_list,
                                         self.args.testims_dir_tot,
                                         self.args.test_splitims_locs_file2,
                                         slice_sizes=self.args.slice_sizes2,
                                         slice_overlap=self.args.slice_overlap,
                                         test_slice_sep=self.args.test_slice_sep,
                                         zero_frac_thresh=self.args.zero_frac_thresh,
                                         )

                df_tot2 = self.run_test(infer_cmd=test_cmd_tot2,
                                        framework=self.args.framework,
                                        results_dir=self.args.results_dir,
                                        log_file=self.args.log_file,
                                        test_files_locs_list=test_files_locs_list2,
                                        slice_sizes=self.args.slice_sizes,
                                        testims_dir_tot=self.args.testims_dir_tot2,
                                        yolt_test_classes_files=self.args.yolt_test_classes_files2,
                                        label_map_dict=self.args.label_map_dict2,
                                        val_df_path_init=self.args.val_df_path_init2,
                                        # val_df_path_aug=self.args.val_df_path_aug2,
                                        test_slice_sep=self.args.test_slice_sep,
                                        edge_buffer_test=self.args.edge_buffer_test,
                                        max_edge_aspect_ratio=self.args.max_edge_aspect_ratio,
                                        test_box_rescale_frac=self.args.test_box_rescale_frac,
                                        rotate_boxes=self.args.rotate_boxes,
                                        test_add_geo_coords=self.args.test_add_geo_coords)

                # save to csv
                df_tot2.to_csv(self.args.val_df_path_aug2, index=False)
                t6 = time.time()
                cmd_time_str = '"Length of time to run test' + ' ' \
                    + str(t6 - t5) + ' seconds\n"'
                self.sig1.emit(cmd_time_str[1:-1])
                os.system('echo ' + cmd_time_str + ' >> ' + self.args.log_file)

                # Update category numbers of df_tot2 so that they aren't the same
                #    as df_tot?  Shouldn't need to since categories are strings

                # Combine df_tot and df_tot2
                df_tot = pd.concat([df_tot, df_tot2])
                test_split_dir_list = test_split_dir_list \
                    + test_split_dir_list2

                # Create new label_map_dict with all categories (done in init_args)

            else:
                pass

            # refine and plot
            t8 = time.time()
            if len(np.append(self.args.slice_sizes, self.args.slice_sizes2)) > 0:
                sliced = True
            else:
                sliced = False
            self.sig1.emit("test data sliced?" + str(sliced))

            # refine for each plot_thresh (if we have detections)
            if len(df_tot) > 0:
                for plot_thresh_tmp in self.args.plot_thresh:
                    self.sig1.emit("Plotting at: " + str(plot_thresh_tmp))
                    groupby = 'Image_Path'
                    groupby_cat = 'Category'
                    df_refine = post_process.refine_df(df_tot,
                                                       groupby=groupby,
                                                       groupby_cat=groupby_cat,
                                                       nms_overlap_thresh=self.args.nms_overlap_thresh,
                                                       plot_thresh=plot_thresh_tmp,
                                                       verbose=False)
                    # make some output plots, if desired
                    if len(self.args.building_csv_file) > 0:
                        building_csv_file_tmp = self.args.building_csv_file.split('.')[0] \
                            + '_plot_thresh_' + str(plot_thresh_tmp).replace('.', 'p') \
                            + '.csv'
                    else:
                        building_csv_file_tmp = ''
                    if self.args.n_test_output_plots > 0:
                        outfiles = post_process.plot_refined_df(df_refine, groupby=groupby,
                                                     label_map_dict=self.args.label_map_dict_tot,
                                                     outdir=self.args.results_dir,
                                                     plot_thresh=plot_thresh_tmp,
                                                     show_labels=bool(
                                                         self.args.show_labels),
                                                     alpha_scaling=bool(
                                                         self.args.alpha_scaling),
                                                     plot_line_thickness=self.args.plot_line_thickness,
                                                     print_iter=5,
                                                     n_plots=self.args.n_test_output_plots,
                                                     building_csv_file=building_csv_file_tmp,
                                                     shuffle_ims=bool(
                                                         self.args.shuffle_val_output_plot_ims),
                                                     verbose=False)
                        self.outputs.emit(outfiles)

                    # geo coords?
                    if bool(self.args.test_add_geo_coords):
                        df_refine, json = add_geo_coords.add_geo_coords_to_df(
                            df_refine,
                            create_geojson=bool(self.args.save_json),
                            inProj_str='epsg:32737', outProj_str='epsg:3857',
                            # inProj_str='epsg:4326', outProj_str='epsg:3857',
                            verbose=False)

                    # save df_refine
                    outpath_tmp = os.path.join(self.args.results_dir,
                                               self.args.val_prediction_df_refine_tot_root_part +
                                               '_thresh=' + str(plot_thresh_tmp) + '.csv')
                    # df_refine.to_csv(self.args.val_prediction_df_refine_tot)
                    df_refine.to_csv(outpath_tmp)
                    self.sig1.emit("Num objects at thresh " +
                                   str(plot_thresh_tmp) + "="+str(len(df_refine)))
                    # save json
                    if bool(self.args.save_json) and (len(json) > 0):
                        output_json_path = os.path.join(self.args.results_dir,
                                                        self.args.val_prediction_df_refine_tot_root_part +
                                                        '_thresh=' + str(plot_thresh_tmp) + '.GeoJSON')
                        json.to_file(output_json_path, driver="GeoJSON")

                cmd_time_str = '"Length of time to run refine_test()' + ' ' \
                    + str(time.time() - t8) + ' seconds"'
                self.sig1.emit(cmd_time_str[1:-1])
                os.system('echo ' + cmd_time_str + ' >> ' + self.args.log_file)

            # remove or zip test_split_dirs to save space
            if len(test_split_dir_list) > 0:
                for test_split_dir_tmp in test_split_dir_list:
                    if os.path.exists(test_split_dir_tmp):
                        # compress image chip dirs if desired
                        if self.args.keep_test_slices:
                            self.sig1.emit("Compressing image chips...")
                            shutil.make_archive(test_split_dir_tmp, 'zip',
                                                test_split_dir_tmp)
                        # remove unzipped folder
                        self.sig1.emit(
                            "Removing test_split_dir_tmp:"+test_split_dir_tmp)
                        # make sure that test_split_dir_tmp hasn't somehow been shortened
                        #  (don't want to remove "/")
                        if len(test_split_dir_tmp) < len(self.args.results_dir):
                            self.sig1.emit(
                                "test_split_dir_tmp too short!!!!:"+test_split_dir_tmp)
                            return
                        else:
                            self.sig1.emit("Removing image chips...")

                            shutil.rmtree(test_split_dir_tmp,
                                          ignore_errors=True)

            cmd_time_str = '"Total Length of time to run test' + ' ' \
                + str(time.time() - t3) + ' seconds\n"'
            self.sig1.emit(cmd_time_str[1:-1])
            os.system('echo ' + cmd_time_str + ' >> ' + self.args.log_file)

        # print ("\nNo honeymoon. This is business.")
        self.sig1.emit(
            "\n\n\nWell, I'm glad we got that out of the way.\n\n\n\n")

        return

    ###############################################################################

    def slice_im(self, image_path, out_name, out_dir, sliceHeight=256, sliceWidth=256,
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
        self.sig1.emit("image.shape:"+str(image.shape))

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
                # self.sig1.emit ("zero_frac", zero_fra
                # skip if image is mostly empty
                if zero_frac >= zero_frac_thresh:
                    if verbose:
                        self.sig1.emit(
                            "Zero frac too high at: "+str(zero_frac))
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

        self.sig1.emit("Num slices: "+str(n_ims) + ", Num non-null slices: "+str(n_ims_nonull) +
                       ", sliceHeight: "+str(sliceHeight)+", sliceWidth: "+str(sliceWidth))
        self.sig1.emit("Time to slice "+image_path+": " +
                       str(time.time()-t0)+" seconds")

        return

    ###############################################################################
    def run(self):

        self.args = self.update_args()

        # check framework
        if self.args.framework.upper() == 'YOLT':
            raise ValueError("self.args.framework must specify YOLT2, YOLT3, or "
                             "a TensorFlow model, not YOLT!")

        train_cmd1, test_cmd_tot, test_cmd_tot2 = self.prep()
        self.execute(train_cmd1, test_cmd_tot, test_cmd_tot2)
