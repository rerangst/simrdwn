from app.label import MainWindow_Label
from simrdwn.data_prep.thread_prep_data import PrepDataThread
from simrdwn.core.thread_simrdwn import SimrdwnThread
from skimage import io
from frontend import YOLO
from keras.models import load_model

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib.pyplot as plt
import sys
import os

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *

from threading import Thread
from pathlib import Path
import numpy as np
from decimal import Decimal
import time
import json
import glob
import cv2
import matplotlib

matplotlib.use('QT5Agg')

# from app.libs.canvas import Canvas

def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def ustr(x):
    '''py2/py3 unicode helper'''

    if sys.version_info < (3, 0, 0):
        from PyQt5.QtCore import QString
        if type(x) == str:
            return x.decode('utf-8')
        if type(x) == type('string'):
            return unicode(x)
        return x
    else:
        return x  # py3


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


class PaintCanvas(QWidget):
    sig1 = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QtGui.QPixmap()
        self.scale = 1.0
        self.max_size = QtCore.QSize(1280, 900)
        self.size = self.max_size

    def setImage(self, impath):
        self.image = QtGui.QPixmap(impath)
        im_size = self.image.size()

        max_display_w = self.max_size.width()
        max_display_h = self.max_size.height()
        r_h = max_display_h / im_size.height()
        r_w = max_display_w / im_size.width()
        if r_h * im_size.width() > max_display_w:
            self.scale = r_w
        else:
            self.scale = r_h
        self.sig1.emit('Setting scale to '+str(self.scale))
        # self.display_w = self.scale * im_size.width()
        # self.display_h = self.scale * im_size.height()
        self.size = QtCore.QSize(
            self.scale * im_size.width(), self.scale * im_size.height())

        self.setFixedSize(self.size)
        # self.sig1.emit(self.scale * im_size.height())
        # self.sig1.emit(self.scale * im_size.width())

    def getSize(self):
        return QtCore.QSize()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.scale(self.scale, self.scale)
        p.drawPixmap(0, 0, self.image)


class Window(QMainWindow):

    def __init__(self, prep_data_obj=None, simrdwn=None):

        # Inherit properies from QMainWindow
        super().__init__()

        # Setting geometry attributes of MainWindow
        self.left = 10
        self.top = 10
        self.width = 720
        self.height = 480
        self.project_num = 0
        self.project_name = "_"
        self.project_list = []
        self.image = QtGui.QImage()
        self.imageData = None

        # Design GUI elements
        self.initUI()

        # Declare Camera, simrdwn instances
        self.thread_prep_data = prep_data_obj
        self.thread_simrdwn = simrdwn
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # GUI elements design
    def initUI(self):

        # Matplotlib figure to draw image frame
        self.figure = plt.figure()

        # Window canvas to host matplotlib figure
        # self.canvas = FigureCanvas(self.figure)
        self.canvas = PaintCanvas()
        self.setCentralWidget(self.canvas)
        self.canvas.sig1.connect(self.on_display_msg)

        # Setting Title and StatusBar
        self.setWindowTitle('Ứng dụng phát hiện đối tượng trên ảnh vệ tinh')
        self.statusBar().showMessage("Trạng thái tác vụ: ")
        
        self.dialogs = list()

        # Setting main menu
        mainMenu = self.menuBar()
        prepMenu = mainMenu.addMenu('Chuẩn bị dữ liệu')
        trainMenu = mainMenu.addAction(
            'Huấn luyện mô hình', self.on_show_train_window)
        detectMenu = mainMenu.addAction(
            'Phát hiện đối tượng', self.on_show_detect_window)
        helpMenu = mainMenu.addMenu('Trợ giúp')

        # Setting child menu
        prepMenu.addAction('Cắt ảnh', self.on_show_prep_sliceim_window)
        prepMenu.addAction('Gán nhãn', self.on_show_prep_label_window)
        prepMenu.addAction('Tạo file tfrecord', self.on_show_prep_prepare_window)

        # Create navigate image buttons Widget
        self.wg_nav = QWidget()
        self.btn_prev = QPushButton('<', self.wg_nav)
        self.btn_next = QPushButton('>', self.wg_nav)
        self.btn_prev.clicked.connect(self.on_prev_im)
        self.btn_next.clicked.connect(self.on_next_im)

        # Create buttons Widget
        self.central_widget = QWidget()

        # Just some buttons
        self.btn_load_im_folder = QPushButton(
            'Chọn thư mục ground_truth', self.central_widget)
        self.btn_load_train_data_dir = QPushButton(
            'Thư mục lưu dữ liệu huấn luyện', self.central_widget)
        self.btn_load_weight = QPushButton(
            'Chọn file trọng số', self.central_widget)
        self.btn_load_cfg = QPushButton(
            'Chọn file config', self.central_widget)
        self.btn_slice = QPushButton(
            'Cắt ảnh', self.central_widget)
        self.btn_label_sliced_im = QPushButton(
            'Gán nhãn', self.central_widget)
        self.btn_prepare_dataset = QPushButton(
            'Tạo file tfrecord', self.central_widget)
        self.btn_train = QPushButton(
            'Huấn luyện', self.central_widget)
        self.btn_load_test_im_folder = QPushButton(
            'Chọn thư mục ảnh test', self.central_widget)
        self.btn_load_class_labels = QPushButton(
            'Tải file class_labels.pbtxt', self.central_widget)
        self.btn_detect = QPushButton(
            'Phát hiện đối tượng', self.central_widget)
        # Events handling
        self.btn_load_im_folder.clicked.connect(self.on_open_imagestore)
        self.btn_load_train_data_dir.clicked.connect(
            self.on_open_train_data_dir)
        self.btn_load_weight.clicked.connect(self.on_open_weight_file)
        self.btn_load_cfg.clicked.connect(self.on_open_cfg_file)
        self.btn_slice.clicked.connect(self.on_slice_data)
        self.btn_label_sliced_im.clicked.connect(self.on_label_sliced_im)
        self.btn_prepare_dataset.clicked.connect(self.on_prepare_dataset)
        self.btn_train.clicked.connect(self.on_train)
        self.btn_load_class_labels.clicked.connect(self.on_open_class_labels)
        self.btn_load_test_im_folder.clicked.connect(self.on_open_test_ims)
        self.btn_detect.clicked.connect(self.on_detect)

        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(self.canvas)
        # add navigate buttons
        vBox_nav = QHBoxLayout()
        vBox_nav.addWidget(self.btn_prev)
        vBox_nav.addWidget(self.btn_next)
        self.left_layout.addLayout(vBox_nav)

        self.right_layout = QVBoxLayout()

        # Put Buttons to HBoxLayout
        vBox = QVBoxLayout()
        vBox.addWidget(self.btn_load_im_folder)
        vBox.addWidget(self.btn_load_weight)
        vBox.addWidget(self.btn_load_cfg)
        vBox.addWidget(self.btn_slice)
        vBox.addWidget(self.btn_load_train_data_dir)
        vBox.addWidget(self.btn_label_sliced_im)
        vBox.addWidget(self.btn_prepare_dataset)
        vBox.addWidget(self.btn_train)
        vBox.addWidget(self.btn_load_class_labels)
        vBox.addWidget(self.btn_load_test_im_folder)
        vBox.addWidget(self.btn_detect)
        # Add hBoxLayout to VBoxLayout
        self.right_layout.addLayout(vBox)

        # Hide all buttons
        self.btn_load_im_folder.setVisible(False)
        self.btn_load_weight.setVisible(False)
        self.btn_load_cfg.setVisible(False)
        self.btn_slice.setVisible(False)
        self.btn_label_sliced_im.setVisible(False)
        self.btn_load_train_data_dir.setVisible(False)
        self.btn_prepare_dataset.setVisible(False)
        self.btn_train.setVisible(False)
        self.btn_load_class_labels.setVisible(False)
        self.btn_load_test_im_folder.setVisible(False)
        self.btn_detect.setVisible(False)

        self.textedit = QTextEdit()
        font = QtGui.QFont()
        font.setPointSize(9)
        self.textedit.setFont(font)
        self.textedit.setFixedWidth(300)
        self.right_layout.addWidget(self.textedit)

        self.layout = QHBoxLayout(self.central_widget)
        self.layout.addLayout(self.left_layout)
        self.layout.addLayout(self.right_layout)

        # Set central widget
        self.setCentralWidget(self.central_widget)

    def on_show_prep_sliceim_window(self):
        """
        Display data generation for training window
        """
        self.btn_load_im_folder.setVisible(True)
        self.btn_load_weight.setVisible(False)
        self.btn_load_cfg.setVisible(False)
        self.btn_slice.setVisible(True)
        self.btn_label_sliced_im.setVisible(False)
        self.btn_load_train_data_dir.setVisible(False)
        self.btn_prepare_dataset.setVisible(False)
        self.btn_train.setVisible(False)
        self.btn_load_class_labels.setVisible(False)
        self.btn_load_test_im_folder.setVisible(False)
        self.btn_detect.setVisible(False)

    def on_show_prep_label_window(self):
        """
        Display data generation for training window
        """
        self.btn_load_im_folder.setVisible(False)
        self.btn_load_weight.setVisible(False)
        self.btn_load_cfg.setVisible(False)
        self.btn_slice.setVisible(False)
        self.btn_load_train_data_dir.setVisible(True)
        self.btn_label_sliced_im.setVisible(True)
        self.btn_prepare_dataset.setVisible(False)
        self.btn_train.setVisible(False)
        self.btn_load_test_im_folder.setVisible(False)
        self.btn_detect.setVisible(False)

    def on_show_prep_prepare_window(self):
        """
        Display data generation for training window
        """
        self.btn_load_im_folder.setVisible(False)
        self.btn_load_weight.setVisible(False)
        self.btn_load_cfg.setVisible(False)
        self.btn_slice.setVisible(False)
        self.btn_label_sliced_im.setVisible(False)
        self.btn_load_train_data_dir.setVisible(True)
        self.btn_prepare_dataset.setVisible(True)
        self.btn_train.setVisible(False)
        self.btn_load_class_labels.setVisible(False)
        self.btn_load_test_im_folder.setVisible(False)
        self.btn_detect.setVisible(False)

    def on_show_train_window(self):
        """
        Display train window
        """
        self.btn_load_im_folder.setVisible(False)
        self.btn_load_weight.setVisible(True)
        self.btn_load_cfg.setVisible(True)
        self.btn_slice.setVisible(False)
        self.btn_label_sliced_im.setVisible(False)
        self.btn_load_train_data_dir.setVisible(True)
        self.btn_prepare_dataset.setVisible(False)
        self.btn_train.setVisible(True)
        self.btn_load_class_labels.setVisible(False)
        self.btn_load_test_im_folder.setVisible(False)
        self.btn_detect.setVisible(False)

    def on_show_detect_window(self):
        self.btn_load_im_folder.setVisible(False)
        self.btn_load_weight.setVisible(False)
        self.btn_load_cfg.setVisible(False)
        # self.btn_load_results_dir.setVisible(True)
        self.btn_slice.setVisible(False)
        self.btn_label_sliced_im.setVisible(False)
        self.btn_load_train_data_dir.setVisible(False)
        self.btn_prepare_dataset.setVisible(False)
        self.btn_train.setVisible(False)
        self.btn_load_class_labels.setVisible(False)
        self.btn_load_test_im_folder.setVisible(True)
        self.btn_detect.setVisible(True)

    @QtCore.pyqtSlot()
    def on_prev_im(self):
        self.current_idx = self.current_idx - 1
        if self.current_idx < 0:
            self.current_idx = len(self.imageList)-1
        self.textedit.append(
            'Reading image '+self.imageList[self.current_idx]+'...')
        self.canvas.setImage(self.imageList[self.current_idx])
        if  len(self.pltImgList)>0:
            img = io.imread(self.pltImgList[self.current_idx])
            plt.imshow(img)
            plt.show()

    @QtCore.pyqtSlot()
    def on_next_im(self):
        self.current_idx = self.current_idx + 1
        if self.current_idx >= len(self.imageList):
            self.current_idx = 0
        self.textedit.append(
            'Reading image '+self.imageList[self.current_idx]+'...')
        self.canvas.setImage(self.imageList[self.current_idx])
        if  len(self.pltImgList)>0:
            img = io.imread(self.pltImgList[self.current_idx])
            plt.imshow(img)
            plt.show()
    # Open image store
    @QtCore.pyqtSlot()
    def on_open_imagestore(self):
        # extract file name in a directory
        self.statusBar().showMessage("Chương trình đang xử lý tác vụ: Mở kho ảnh")
        filepath = QFileDialog().getExistingDirectory(
            self, "Chọn đường dẫn tới kho ảnh")
        if filepath:
            self.statusBar().showMessage(str("Kho ảnh nằm tại: ") + filepath)
            train_dir = filepath+'/train'
            test_dir = filepath+'/test'
            label_map_path = filepath+'/class_labels.pbtxt'
            self.textedit.append('Train folder: '+train_dir)
            self.textedit.append('Test folder: '+test_dir)

            """ Plot some samples from train folder """
            exts = ['*.png', '*.jpg']
            self.imageList = [f for ext in exts for f in glob.glob(
                os.path.join(train_dir, ext))]
            # print(self.imageList)

            # check filename list and display first image
            if len(self.imageList) <= 0:
                self.textedit.append(
                    '[ERROR] No images found in train dir ('+train_dir+')')
            else:
                self.thread_prep_data.ground_truth_dir = filepath
                self.thread_prep_data.label_map_path = label_map_path
                # self.thread_prep_data.project_name = 'Project #'+str(self.project_num)

                self.current_idx = 0
                self.textedit.append(
                    'Reading image '+self.imageList[self.current_idx]+'...')

                self.canvas.setImage(self.imageList[self.current_idx])
                # self.canvas.paintEvent()

                # self.imageData = read(self.imageList[self.current_idx], None)
                # image = QtGui.QImage.fromData(self.imageData)
                # self.image = image
                # self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
                # self.setClean()
                # self.canvas.setEnabled(True)
                # self.adjustScale(initial=True)
                # self.paintCanvas()

                # img = cv2.imread(self.imageList[self.current_idx])
                # pic = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # plt.imshow(pic)
                # self.canvas.draw()
        else:
            self.statusBar().showMessage(filepath)

    @QtCore.pyqtSlot()
    def on_open_train_data_dir(self):
        """ Open train data dir """
        # extract file name in a directory
        filepath = QFileDialog().getExistingDirectory(
            self, "Path to train data directory")
        if filepath:
            images_dir = filepath+'/images'
            # labels_dir = filepath+'/labels'
            labels_yolo_dir = filepath+'/labels_yolo'
            label_map_path = filepath+'/class_labels.pbtxt'
            self.textedit.append('Images folder: '+images_dir)
            self.textedit.append(
                'Labels folder (in yolo format): '+labels_yolo_dir)
            self.textedit.append('label_map_path: '+label_map_path)

            self.thread_prep_data.train_out_dir = filepath
            # self.thread_prep_data.label_map_path = label_map_path
        else:
            self.statusBar().showMessage(filepath)

    def on_display_msg(self, msg):
        """ Display messages from threads """
        self.textedit.append(msg)

    @QtCore.pyqtSlot()
    def on_slice_data(self):
        self.thread_prep_data.update_args(
            ground_truth_dir=self.thread_prep_data.ground_truth_dir)

        self.statusBar().showMessage("Chương trình đang xử lý tác vụ: Tiền xử lý dữ liệu - ")
        # self.textedit.clear()
        self.textedit.append("")
        self.textedit.append("***********************************")
        # self.textedit.append("")
        self.textedit.append(str("Tiền xử lý dữ liệu huấn luyện"))

        # self.thread_prep_data = PrepDataThread()
        self.thread_prep_data.sig1.connect(
            self.on_display_msg)
        # self.thread_prep_data.sig2.connect(
        #     self.on_image_preprocessed_display)
        # self.thread_prep_data.sig3.connect(
        #     self.on_image_preprocessed_result)
        self.thread_prep_data.action = 'slice'
        self.thread_prep_data.start()

    @QtCore.pyqtSlot()
    def on_label_sliced_im(self):
        self.thread_prep_data.train_out_dir = 'data/train_data'

        self.thread_prep_data.update_args(
            train_out_dir=self.thread_prep_data.train_out_dir)
        self.statusBar().showMessage("Chương trình đang xử lý tác vụ: Tiền xử lý dữ liệu - ")
        # self.textedit.clear()
        self.textedit.append("")
        self.textedit.append("***********************************")
        # self.textedit.append("")
        self.textedit.append(str("Tiền xử lý dữ liệu huấn luyện"))

        self.thread_prep_data.action = 'label'
        self.thread_prep_data.start()
        
        dialog = MainWindow_Label(self, defaultFilename=self.thread_prep_data.train_out_dir,
                                  defaultPrefdefClassFile=self.thread_prep_data.train_out_dir+'/classes.txt')
        dialog.sig_action.connect(self.on_receive_action_signal)
        self.dialogs.append(dialog)
        dialog.show()

    def on_receive_action_signal(self, action):
        print('action', action)
        if action == 'convert_voc2yolo':
            self.thread_prep_data.action = action
            self.thread_prep_data.sig1.connect(self.on_display_msg)
            self.thread_prep_data.start()

    @QtCore.pyqtSlot()
    def on_prepare_dataset(self):
        self.thread_prep_data.update_args(
            train_out_dir=self.thread_prep_data.train_out_dir)
        self.statusBar().showMessage("Chương trình đang xử lý tác vụ: Tiền xử lý dữ liệu")
        # self.textedit.clear()
        self.textedit.append("")
        self.textedit.append("***********************************")
        # self.textedit.append("")
        self.textedit.append(str("Tiền xử lý dữ liệu huấn luyện"))

        # self.thread_prep_data = PrepDataThread()
        self.thread_prep_data.sig1.connect(
            self.on_display_msg)
        # self.thread_prep_data.sig2.connect(
        #     self.on_image_preprocessed_display)
        # self.thread_prep_data.sig3.connect(
        #     self.on_image_preprocessed_result)
        self.thread_prep_data.action = 'prepare_dataset'
        self.thread_prep_data.start()

    def on_display_msg(self, msg):
        """ Display messages from PrepDataThread """
        self.textedit.append(msg)

    @QtCore.pyqtSlot()
    def on_train(self):
        self.statusBar().showMessage("Chương trình đang xử lý tác vụ: Tiền xử lý dữ liệu")
        # self.textedit.clear()
        self.textedit.append("")
        self.textedit.append("***********************************")
        # self.textedit.append("")
        self.textedit.append(str("Huấn luyện mô hình"))

        # self.thread_simrdwn = PrepDataThread()
        self.thread_simrdwn.sig1.connect(
            self.on_display_msg)
        # self.thread_prep_data.sig2.connect(
        #     self.on_image_preprocessed_display)
        # self.thread_prep_data.sig3.connect(
        #     self.on_image_preprocessed_result)
        self.thread_simrdwn.args.mode = 'train'
        self.thread_simrdwn.args.train_data_dir = self.thread_prep_data.train_out_dir
        self.thread_simrdwn.start()

    @QtCore.pyqtSlot()
    def on_open_weight_file(self):
        """ Open weight file """
        filters = "Open weight file (%s)" % ' '.join(['*.weights'])
        filepath = ustr(QFileDialog().getOpenFileName(
            self, "Path to input weight", filters))
        if filepath:
            if isinstance(filepath, (tuple, list)):
                filepath = filepath[0]

            self.textedit.append('Input weight: '+filepath)

            self.thread_simrdwn.args.weight_file_tot = filepath
            self.thread_simrdwn.args.weight_file = filepath.split('/')[-1]
            self.thread_simrdwn.args.yolt_weight_dir = filepath.split(
                self.thread_simrdwn.args.weight_file)[0]
            # self.thread_prep_data.label_map_path = label_map_path
        else:
            self.statusBar().showMessage(filepath)

    @QtCore.pyqtSlot()
    def on_open_cfg_file(self):
        """ Open cfg file """
        filters = "Open model cfg file (%s)" % ' '.join(['*.cfg'])
        filepath = ustr(QFileDialog().getOpenFileName(
            self, "Path to model cfg file", filters))
        if filepath:
            if isinstance(filepath, (tuple, list)):
                filepath = filepath[0]

            self.textedit.append('Model cfg: '+filepath)

            self.thread_simrdwn.args.yolt_cfg_file_in = filepath
            self.thread_simrdwn.args.yolt_cfg_file = filepath.split('/')[-1]
            self.thread_simrdwn.args.yolt_cfg_dir = filepath.split(
                self.thread_simrdwn.args.yolt_cfg_file)[0]
            # self.thread_prep_data.label_map_path = label_map_path
        else:
            self.statusBar().showMessage(filepath)

    @QtCore.pyqtSlot()
    def on_open_class_labels(self):
        """ Open class_labels file """
        filters = "Open class_labels file (%s)" % ' '.join(['*.cfg'])
        filepath = ustr(QFileDialog().getOpenFileName(
            self, "Path to class_labels.pbtxt file", filters))
        if filepath:
            if isinstance(filepath, (tuple, list)):
                filepath = filepath[0]

            self.textedit.append('label_map_path: '+filepath)

            self.thread_simrdwn.args.label_map_path = filepath
            # self.thread_prep_data.label_map_path = label_map_path
        else:
            self.statusBar().showMessage(filepath)

    @QtCore.pyqtSlot()
    def on_detect(self):
        self.statusBar().showMessage("Chương trình đang xử lý tác vụ: Đánh giá mô hình ")
        # self.textedit.clear()
        self.textedit.append("")
        self.textedit.append("***********************************")
        # self.textedit.append("")
        self.textedit.append(str("Phát hiện đối tượng"))
        with open('abc/config_plane.json') as config_buffer:
            config = json.load(config_buffer)
        yolo = YOLO(backend=config['model']['backend'],
                    input_size=config['model']['input_size'],
                    labels=config['model']['labels'],
                    max_box_per_image=config['model']['max_box_per_image'],
                    anchors=config['model']['anchors'])
        yolo.load_weights('full_yolo_plane.h5')
        classModel = load_model('model_saved1.h5')
        obj_threshold=0.3
        for imgPath in self.imageList:
            WINDOW_SIZE = (544, 544)
            STRIDE = 408
            img = io.imread(imgPath)
            print(imgPath ," Shape: ", img.shape)
            print("Total pieces images: ",count_sliding_window(img, step=STRIDE, window_size=WINDOW_SIZE))
            listRects=[]
            for y, x, ws_0, ws_1 in sliding_window(img, step=STRIDE, window_size=WINDOW_SIZE):
                image = img[y:y+ws_1,x:x+ws_0]
                boxes = yolo.predict(image,obj_threshold=obj_threshold)
                if  len(boxes)>0:
                    # print(len(boxes), 'boxes are found')
                    image_h, image_w, _ = image.shape
                    for box in boxes:
                        xmin = int(box.xmin * image_w)
                        ymin = int(box.ymin * image_h)
                        xmax = int(box.xmax * image_w)
                        ymax = int(box.ymax * image_h)
                        srcXmin=x+ xmin
                        srcXmax=x+ xmax
                        srcYmin=y+ ymin
                        srcYmax=y+ ymax
                        if srcXmax>0 and srcXmin>0 and srcYmin>0 and srcYmax>0 \
                                and (srcXmax-srcXmin)>40 and (srcYmax-srcYmin)>40:
                            preImg= img[srcYmin:srcYmax,srcXmin:srcXmax]
                            preImg=cv2.resize(preImg,(80,80))
                            input = np.expand_dims(preImg, axis=0)
                            if classModel.predict(input)[0] == 1:
                                # num = num + 1
                                listRects.append((srcXmin , srcXmax, srcYmin, srcYmax))
            for i, rect in enumerate(listRects):
                x0,x1,y0,y1= rect
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 3)
            savePath=imgPath[:-4] + 'detected' + '.jpg'
            cv2.imwrite(savePath,img)
            self.textedit.append(str.format(imgPath," Total ",len(listRects)," boxes detected!!!!!!!!!!"))
            print(" Total ",len(listRects)," boxes detected!!!!!!!!!!")
            self.pltImgList.append(savePath)
            # plt.imshow(img)
            # plt.show()


    def on_display_result_image(self, outfiles):
        print('outfiles', outfiles)
        self.imageList = outfiles
        self.textedit.append('Showing detection results...')
        self.canvas.setImage(self.imageList[self.current_idx])

    @QtCore.pyqtSlot()
    def on_open_test_ims(self):
        # extract file name in a directory
        self.statusBar().showMessage("Chương trình đang xử lý tác vụ: Mở kho ảnh")
        filepath = QFileDialog().getExistingDirectory(
            self, "Chọn đường dẫn tới kho ảnh")
        if filepath:
            self.statusBar().showMessage(str("Kho ảnh nằm tại: ") + filepath)
            self.textedit.append('Test images: '+filepath)

            """ Plot some samples from train folder """
            exts = ['*.png', '*.jpg']
            self.imageList = [f for ext in exts for f in glob.glob(
                os.path.join(filepath, ext))]
            # print(self.imageList)
            self.pltImgList=[]
            # check filename list and display first image
            if len(self.imageList) <= 0:
                self.textedit.append(
                    '[ERROR] No images found in directory ('+filepath+')')
            else:
                self.thread_simrdwn.args.testims_dir_tot = filepath
                # self.thread_prep_data.project_name = 'Project #'+str(self.project_num)
                self.textedit.append(str(len(self.imageList))+' test images')

                self.current_idx = 0
                self.textedit.append(
                    'Reading image '+self.imageList[self.current_idx]+'...')

                self.canvas.setImage(self.imageList[self.current_idx])

                # img = cv2.imread(self.imageList[self.current_idx])
                # pic = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # plt.imshow(pic)
                # self.canvas.draw()
        else:
            self.statusBar().showMessage(filepath)


# Main program
if __name__ == '__main__':
    app = QApplication([])
    prep_data_thread = PrepDataThread()
    # Initialize new Simrdwn object
    simrdwn = SimrdwnThread()
    # Create MainWindow with objects
    main = Window(prep_data_thread, simrdwn)
    main.show()

    app.exit(app.exec_())
