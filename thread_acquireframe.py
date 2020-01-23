import sys, os
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
from threading import Thread
from pathlib import Path
import numpy as np
from decimal import Decimal
import time
import glob
import cv2
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar



class Webcam:
  
    def __init__(self, cam_num):
        #self.video_capture = cv2.VideoCapture(0)
        self.cam_num = cam_num
        self.cap = None
        self.current_state = False
        #self.current_frame = self.video_capture.read()[1]
        self.current_frame = np.zeros((1,1))

    def initialize(self):
        self.cap = cv2.VideoCapture(self.cam_num)
          
    #create thread for capturing images
    def start(self):
        Thread(target=self._update_frame, args=()).start()
  
    def _update_frame(self):
        #while(True):
        ret, frame = self.cap.read()
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_state = True
                
    #get the current frame
    def get_current_frame(self):
        return self.current_frame

class Calibration:
    
    def __init__(self):
        
        self.project_name = "Calib_"
        #Image files path
        self.imagelist = []
        self.last_processed_image = ''
        self.validated_image_index = []
        self.gray_inv_shape = ()
        self.preprocessing_done = False
        self.processed_count = 0
        self.validated_count = 0
        self.image_name = []
        self.validated_image_name = []
        self.calibrated = False
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.mean_eror = -1
        self.report = []

        #termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #prepare object points, like (0,0,0), (1,0,0), (2,0,0),..., (6,5,0)
        self.objp = np.zeros((6*7,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)
        #Arrays to store object points and image points from all the images
        self.objpoints = [] #3d points in real world space
        self.imgpoints = [] #2d points in image plane

    def refresh(self):
        self.project_name = "Calib_"
        #Image files path
        #calib.imagelist = self.imagelist
        self.last_processed_image = ''
        self.validated_image_index = []
        self.gray_inv_shape = ()
        self.preprocessing_done = False
        self.processed_count = 0
        self.validated_count = 0
        self.image_name = []
        self.validated_image_name = []
        self.calibrated = False
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.mean_eror = -1
        self.report = []
        #termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #prepare object points, like (0,0,0), (1,0,0), (2,0,0),..., (6,5,0)
        self.objp = np.zeros((6*7,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)
        #Arrays to store object points and image points from all the images
        self.objpoints = [] #3d points in real world space
        self.imgpoints = [] #2d points in image plane

    def clone(self):
        calib = Calibration()
        calib.project_name = self.project_name
        #Image files path
        calib.imagelist = self.imagelist
        caliblast_processed_image = self.last_processed_image
        calib.validated_image_index = self.validated_image_index
        calib.gray_inv_shape = self.gray_inv_shape
        calib.preprocessing_done = self.preprocessing_done
        calib.processed_count = self.processed_count
        calib.validated_count = self.validated_count
        calib.image_name = self.image_name
        calib.validated_image_name = self.validated_image_name
        calib.calibrated = self.calibrated
        calib.mtx = self.mtx
        calib.dist = self.dist
        calib.rvec = self.rvecs
        calib.tvec = self.tvecs
        calib.mean_eror = self.mean_eror
        calib.report = self.report
        #Arrays to store object points and image points from all the images
        calib.objpoints = self.objpoints #3d points in real world space
        calib.imgpoints = self.imgpoints #2d points in image plane
        return calib

    def process_image(self, fname):
        #Convert fname to string, extract image name, add to a list
        fnames = str(fname)
        name = fnames.split('\\')[-1]
        self.image_name.append(name)        
        #Replace sigle backslash to double backslash
        filename = fnames.replace('\\','\\\\')        
        # Read, convert image to gray
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pic = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #find the chess board corners
        corners = []
        corners2 = []
        ret = False
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)               
        #If found, add object points, image points (after refining them)
        if ret == True:
            self.objpoints.append(self.objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
            self.imgpoints.append(corners2)
            cv2.drawChessboardCorners(pic, (7,6), corners2, ret)
            self.last_processed_image = name
            self.validated_image_name.append(name)
            self.validated_count += 1
            self.validated_image_index.append(self.processed_count)            
        self.processed_count += 1
        if self.processed_count == len(self.imagelist):
            self.preprocessing_done = True        
        #Return ret and pic
        return ret, pic

    def calib(self):
        if (self.calibrated == False) and (self.validated_count > 0):
            #Calculate shape
            fname = str(self.imagelist[int(self.validated_image_index[0])])
            filename = fname.replace('\\','\\\\')
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_inv_shape = gray.shape[::-1]
            self.gray_inv_shape = gray_inv_shape
            #Calibrate
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray_inv_shape, None, None)
            self.calibrated = True
            #Mean eror calculation
            self.error_reprojected()
            # Calibration report
            self.report = self.calibration_report()    
        return ret, self.mtx, self.dist, self.rvecs, self.tvecs

    def error_reprojected(self):
        self.mean_error = -1
        if self.calibrated == True:
            self.mean_error = 0
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                self.mean_error += error
            self.mean_error /= len(self.objpoints)
        return self.mean_error

    def calibration_report(self):
        report = []
        if self.calibrated ==True:
            report.append(str("Số ảnh đã xử lý - tổng số ") + str(self.processed_count) + str(" ảnh "))            
            report.append(str("Số ảnh hợp lệ cho kiểm chuẩn - tổng số ") + str(self.validated_count) + str(" ảnh "))
            report.append(str("Tổng số điểm ảnh tham gia kiểm chuẩn: ") + str(len(self.imgpoints)*len(self.imgpoints[0])))
            report.append("")
            report.append("Kết quả kiểm chuẩn:")
            report.append("  1. Ma trận máy ảnh (Đơn vị - pixel) ")
            report.append("  2. Các hệ số biến dạng k1, k2, p1, p2, k3 ")
            report.append("  3. Các vetor tư thế chụp ảnh ( Đơn vị - độ) ")
            report.append("  4. Các vector khoảng cách chụp ảnh ( Đơn vị - mét) ")
            report.append(str("  5. Sai số trung phương chiếu ngược từ ") + str(len(self.imgpoints)*len(self.imgpoints[0])) + str(" điểm ảnh"))
            report.append("")            
            mtx = self.mtx
            dist = self.dist
            rvecs = self.rvecs
            tvecs = self.tvecs
            report.append("1. Ma trận máy ảnh (Đơn vị - pixel): ")
            report.append(str(mtx))
            report.append(" ")
            report.append("2. Các hệ số biến dạng k1, k2, p1, p2, k3: ")
            dist_vec = dist[0]
            report.append(str("   k1 = ") + str(dist_vec[0]))
            report.append(str("   k2 = ") + str(dist_vec[1]))
            report.append(str("   p1 = ") + str(dist_vec[2]))
            report.append(str("   p2 = ") + str(dist_vec[3]))
            report.append(str("   k3 = ") + str(dist_vec[4]))
            report.append(str(" "))
            report.append(str("3. Các vetor tư thế chụp ảnh ( Đơn vị - độ): "))            
            rad_deg = 180/3.14
            index = 0
            for rvec in rvecs:
                rvec = np.array(rvec)
                rvec = rvec.reshape(-1,3)
                rvec = rvec[0]
                rvec[0] = round(rvec[0]*rad_deg, 4)
                rvec[1] = round(rvec[1]*rad_deg, 4)
                rvec[2] = round(rvec[2]*rad_deg, 4)
                report.append(str("   Tư thế chụp ảnh ") + str(self.validated_image_name[index]) + str(": ")+ str(tuple(rvec)))
                index += 1
            report.append(" ")
            report.append("4. Các vector khoảng cách chụp ảnh ( Đơn vị - mét): ")            
            index = 0
            for tvec in tvecs:
                tvec = np.array(tvec)
                tvec = tvec.reshape(-1,3)
                tvec = tvec[0]
                tvec[0] = round(tvec[0], 4)
                tvec[1] = round(tvec[1], 4)
                tvec[2] = round(tvec[2], 4)
                report.append(str("   Khoảng cách chụp ảnh ") + str(self.validated_image_name[index]) + str(": ")+ str(tuple(tvec)))
                index += 1
            report.append("")            
            report.append(str("5. Sai số trung phương chiếu ngược từ ") + str(len(self.imgpoints)*len(self.imgpoints[0])) + str(" điểm vật (Đơn vị - pixel)"))
            report.append(str("   Mean error: ") + str(self.mean_error))
        return report
            
        
        
class Window(QMainWindow):

    def __init__(self, camera = None, calibration = None):

        #Inherit properies from QMainWindow
        super().__init__()

        #Setting geometry attributes of MainWindow
        self.left = 10
        self.top = 10
        self.width = 720
        self.height = 480
        self.project_num = 0
        self.project_name = "Calib_"
        self.project_list = []

        #Design GUI elements
        self.initUI()

        #Declare Camera, calibration instances 
        self.camera = camera
        self.calib = calibration
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
    #GUI elements design   
    def initUI(self):
        
        # Matplotlib figure to draw image frame
        self.figure = plt.figure()
        
        # Window canvas to host matplotlib figure
        self.canvas = FigureCanvas(self.figure)
        
        # ToolBar to navigate image frame
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.hide()
        
        #Setting Title and StatusBar
        self.setWindowTitle('Ứng dụng kiểm chuẩn máy ảnh số phổ thông')
        self.statusBar().showMessage("Trạng thái tác vụ: ")
        
        #Setting main menu
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('Thao tác tệp tin')
        toolsMenu = mainMenu.addMenu('Xử lý dữ liệu')        
        viewMenu = mainMenu.addMenu('Xem thông tin')
        editMenu = mainMenu.addMenu('Biên tập')
        searchMenu = mainMenu.addMenu('Tìm kiếm')        
        helpMenu = mainMenu.addMenu('Trợ giúp')

        #Setting fileMenu        
        fileMenu.addAction('Thiết lập dự án mới', self.on_file_setnewproject)
        fileMenu.addAction('Mở kho ảnh', self.on_open_imagestore)

        #Setting ToolsMenu        
        toolsMenu.addAction('Tiền xử lý ảnh', self.on_preprocessing_image)
        toolsMenu.addAction('Kiểm chuẩn máy ảnh', self.on_calibrate_image)

        #Setting viewMenu
        viewMenu.addAction('Thông tin dự án', self.on_view_project_info)
        viewMenu.addAction('Dữ liệu kiểm chuẩn', self.on_view_calib_info)
        
        # Create central Widget
        self.central_widget = QWidget()        
        
        # Just some button 
        self.button = QPushButton('Chụp ảnh',self.central_widget)
        self.button1 = QPushButton('Phóng ảnh',self.central_widget)
        self.button2 = QPushButton('Điều hướng ảnh',self.central_widget)
        self.button3 = QPushButton('Về gốc',self.central_widget)
        
        #Put Buttons to HBoxLayout
        hBox = QHBoxLayout()
        hBox.addWidget(self.button)
        hBox.addWidget(self.button1)
        hBox.addWidget(self.button2)
        hBox.addWidget(self.button3)
        
        # set QVBoxLayout as the layout
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        
        #Add hBoxLayout to VBoxLayout
        self.layout.addLayout(hBox)

        self.textedit = QTextEdit()
        font = QtGui.QFont()
        font.setPointSize(9)
        self.textedit.setFont(font)
        self.layout.addWidget(self.textedit)
        
        #Set central widget
        self.setCentralWidget(self.central_widget)
                
        # Events handling
        self.button.clicked.connect(self.plot)
        self.button1.clicked.connect(self.zoom)
        self.button2.clicked.connect(self.pan)
        self.button3.clicked.connect(self.home)

    #Events processing methods

    # Open image store
    @QtCore.pyqtSlot()
    def  on_open_imagestore(self):
        #extract file name in a directory
        self.statusBar().showMessage("Chương trình đang xử lý tác vụ: Mở kho ảnh")        
        filepath = QFileDialog().getExistingDirectory(self, "Chọn đường dẫn tới kho ảnh")
        if filepath:
            images = Path(filepath).glob('*.jpg')
            imagelist = [str(image) for image in images]            
            self.statusBar().showMessage(str("Kho ảnh nằm tại: ") + filepath)
            # check filename list and display first image
            if len(imagelist) > 2:
                #self.calib.imagelist = imagelist
                if self.project_num == 0:                    
                    self.project_num += 1
                    self.calib.imagelist = imagelist
                    self.calib.project_name = str("Calib_") + str(self.project_num)
                else:                                        
                    self.calib.project_name = str("Calib_") + str(self.project_num)
                    self.project_list.append(self.calib)
                    self.project_num += 1
                    newcalib = Calibration()
                    newcalib.project_num = self.project_num
                    newcalib.imagelist = imagelist
                    self.calib = newcalib
                    
                img = cv2.imread(imagelist[0])
                pic = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(pic)
                self.canvas.draw()
            else:
                self.statusBar().showMessage("Không có ảnh trong thư mục được chọn!")
        else:
            self.statusBar().showMessage(filepath)

    #Setting info for new project    
    @QtCore.pyqtSlot()
    def on_file_setnewproject(self):
        self.statusBar().showMessage("Chương trình đang xử lý tác vụ: Thiết lập tham số cho dự án mới") 

    #Preprocessing images of current project
    @QtCore.pyqtSlot()
    def on_preprocessing_image(self):
        
        if len(self.calib.imagelist) > 0:
            self.calib.refresh()
            self.statusBar().showMessage("Chương trình đang xử lý tác vụ: Tiền xử lý ảnh - ")
            #self.textedit.clear()
            self.textedit.append("")
            self.textedit.append("***********************************")
            #self.textedit.append("")
            self.textedit.append(str("Tiền xử lý ảnh dự án Calib_") + str(self.project_num) + str(":") )
            self.thread_image_preprocessing = Calib_Preprocess_Thread(self.calib)
            self.thread_image_preprocessing.sig1.connect(self.on_image_preprocessing_display)
            self.thread_image_preprocessing.sig2.connect(self.on_image_preprocessed_display)
            self.thread_image_preprocessing.sig3.connect(self.on_image_preprocessed_result)
            self.thread_image_preprocessing.start()

    #Calibrate current project
    @QtCore.pyqtSlot()
    def on_calibrate_image(self):
        if self.calib.preprocessing_done == True:
            # Calibrate if calib has not done
            if self.calib.calibrated == False:
                self.thread_calib = Calib_Calibrate_Thread(self.calib)
                self.thread_calib.sig1.connect(self.on_calibtation_processed)
                self.thread_calib.start()
        

    @QtCore.pyqtSlot()
    def on_view_project_info(self):
        pass    

    @QtCore.pyqtSlot()
    def on_view_calib_info(self):
        calib_list = []
        if len(self.project_list) > 0:
            for calib in self.project_list:
                clone_calib = calib.clone()
                calib_list.append(clone_calib)
                
        if self.calib.calibrated == True:        
            clone_calib = self.calib.clone()
            clone_calib.project_name = str("Calib_") + str(self.project_num)
            calib_list.append(clone_calib)

        if len(calib_list) > 0:
            self.calib_info = Total_Calib_Info(calib_list)
            self.calib_info.show()            
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Chưa có dự án kiểm chuẩn được thực hiện hoàn chỉnh!")
            msg.setWindowTitle("Thông báo")
            msg.exec_()
            
            
    def home(self):
        self.toolbar.home()

    def zoom(self):
        self.toolbar.zoom(0.5)

    def pan(self):
        self.toolbar.pan()
         
    def plot(self):
        #Moving Camera class to a thread and emit signal from that thread            
        self.thread1 = Frame_Capture_Thread(self.camera)
        self.thread1.sig1.connect(self.on_frame)
        self.thread1.start()
        
    def on_frame(self, frame):
        plt.imshow(frame)
        self.canvas.draw()

    #Update current image name being processed: fname is image name obtained from signal sig1 of Calib_Preprocess_Thread
    def on_image_preprocessing_display(self, fname):
        filename = fname.split('\\')[-1]
        filename_message = str("Chương trình đang xử lý tác vụ: Tiền xử lý ảnh - đang xử lý file ảnh: ") + filename
        self.statusBar().showMessage(filename_message)
        self.textedit.append(str("Tiền xử lý ảnh - đang xử lý file ảnh: ") + str(filename) + str("..."))
        
    #Update last processed image  on display : pic is image data obtained from signal sig2 of Calib_Preprocess_Thread
    def on_image_preprocessed_display(self, pic):        
        plt.imshow(pic)
        self.canvas.draw()

    #Update preprocessing result -
    def on_image_preprocessed_result(self, result):
        redColor = QtGui.QColor(255,0,0)
        blackColor = QtGui.QColor(0,0,0)
        ret = result[0]
        fname = result[1]
        filename = fname.split('\\')[-1]
        if ret == True:
            self.textedit.setTextColor(blackColor)
            msg = str("Xử lý xong ảnh ") + filename + str(" - ảnh hợp lệ cho kiểm chuẩn!")
            self.textedit.append(msg)
        else:
            self.textedit.setTextColor(redColor)
            msg = str("Xử lý xong ảnh ") + filename + str(" - ảnh không hợp lệ cho kiểm chuẩn!")
            self.textedit.append(msg)
            self.textedit.setTextColor(blackColor)

        if self.calib.preprocessing_done == True:
            msg = str("Tổng số ảnh tiền xử lý: ") + str(self.calib.processed_count)
            self.textedit.append(msg)
            msg = str("Số ảnh hợp lệ cho kiểm chuẩn: ") + str(self.calib.validated_count)
            self.textedit.append(msg)
            msg = str("Số ảnh không hợp lệ cho kiểm chuẩn: ") + str(self.calib.processed_count - self.calib.validated_count)
            self.textedit.append(msg)
            self.statusBar().showMessage("Hoàn thành tác vụ tiền xử lý ảnh - dự án Calib_" + str(self.project_num))
                
    def on_calibtation_processed(self):
        self.calib.project_name = str("Calib_") + str(self.project_num)
        self.calib_info_window = Calib_Info(self.calib)
        self.calib_info_window.show()
        

#QThread for processing slow methods
class Frame_Capture_Thread(QtCore.QThread):
    
    sig1 = QtCore.pyqtSignal(object)    
    def __init__(self, camera):
        super().__init__()
        self.camera = camera

    def run(self):
        self.camera._update_frame()
        self.sig1.emit(camera.current_frame)


class Calib_Preprocess_Thread(QtCore.QThread):

    sig1 = QtCore.pyqtSignal(str)
    sig2 = QtCore.pyqtSignal(object)
    sig3 = QtCore.pyqtSignal(list)
    
    def __init__(self, calibration):
        super().__init__()
        self.calib = calibration

    def run(self):
        if len(self.calib.imagelist) > 0:
            for fname in self.calib.imagelist:
                self.sig1.emit(fname)
                time.sleep(1)
                ret, pic = self.calib.process_image(fname)
                result = []
                result.append(ret)
                result.append(fname)
                self.sig3.emit(result)
                time.sleep(1)
                if ret == True:
                    self.sig2.emit(pic)


class Calib_Calibrate_Thread(QtCore.QThread):

    sig1 = QtCore.pyqtSignal(object)
 
    def __init__(self, calibration):
        super().__init__()
        self.calib = calibration

    def run(self):
        if self.calib.calibrated == False:
            # Calibrate
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = self.calib.calib()
        self.sig1.emit(self.calib) 
        


class Calib_Info(QMainWindow):
    
    def __init__(self, calib = None):
        super(Calib_Info, self).__init__()
        self.calib = calib
        self.initUI()
                
    def initUI(self):   
        layout = QVBoxLayout()
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.textedit = QTextEdit()
        font = QtGui.QFont()
        font.setPointSize(9)
        self.textedit.setFont(font)
        layout.addWidget(self.textedit)
        self.button = QPushButton('Đóng', centralWidget)
        self.button.clicked.connect(self.close_calib_info)
        layout.addWidget(self.button)
        self.setWindowTitle("Dữ liệu kiểm chuẩn máy ảnh")
        self.setGeometry(300,100,700,800)
        self.textedit.append('')
        self.textedit.append('****************************************')
        #self.textedit.append('')
        self.textedit.append(str("Tên dự án: ") + str(self.calib.project_name))
        for rec in self.calib.report:
            self.textedit.append(rec)

    def close_calib_info(self):
        self.close()


class Total_Calib_Info(QMainWindow):
    
    def __init__(self, calib_list = None):
        super(Total_Calib_Info, self).__init__()
        self.calib_list = calib_list
        self.initUI()
                
    def initUI(self):   
        layout = QVBoxLayout()
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.textedit = QTextEdit()
        font = QtGui.QFont()
        font.setPointSize(9)
        self.textedit.setFont(font)
        layout.addWidget(self.textedit)
        self.button = QPushButton('Đóng', centralWidget)
        self.button.clicked.connect(self.close_calib_info)
        layout.addWidget(self.button)
        self.setWindowTitle("Dữ liệu kiểm chuẩn máy ảnh")
        self.setGeometry(300,100,700,800)
        for calib in self.calib_list:
            if calib.calibrated == True:
                self.textedit.append('')
                self.textedit.append('****************************************')
                #self.textedit.append('')
                self.textedit.append(str("Tên dự án: ") + str(calib.project_name))
                for rec in calib.report:
                    self.textedit.append(rec)

    def close_calib_info(self):
        self.close()


     
#Main program
if __name__ == '__main__':
    app = QApplication([])
    #Initialize new Camera object
    camera = Webcam(0)
    camera.initialize()
    #Initialize new Calibration object
    calibration = Calibration()
    #Create MainWindow with objects
    main = Window(camera, calibration)
    main.show()
 
    app.exit(app.exec_())
