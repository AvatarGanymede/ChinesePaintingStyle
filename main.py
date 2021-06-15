import os
import sys
import threading
import time

import cv2 as cv
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from gui.chinesepaintings import Ui_MainWindow
from models import create_model
from edge.hed import extract_edge
import warnings
from utils.params import opt
from utils import create_dataset, html
from utils.visualizer import save_images

warnings.filterwarnings("ignore")
StyleSheet = """
/*标题栏*/
TitleBar {
    background-color: red;
}
/*最小化最大化关闭按钮通用默认背景*/
#buttonMinimum,#buttonClose {
    border: none;
    background-color: red;
}
/*悬停*/
#buttonMinimum:hover{
    background-color: red;
    color: white;
}
#buttonClose:hover {
    color: white;
}
/*鼠标按下不放*/
#buttonMinimum:pressed{
    background-color: Firebrick;
}
#buttonClose:pressed {
    color: white;
    background-color: Firebrick;
}
"""


def remove(path):
    filelist = os.listdir(path)  # 打开对应的文件夹

    for item in filelist:
        os.remove(path+'/'+item)


def getImage(img, self):
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    width = img.shape[1]
    height = img.shape[0]
    image = QImage(img, width, height, QImage.Format_RGB888)
    scale_factor = 0
    if width > height:
        scale_factor = self.org_image.width() / float(width)
    else:
        scale_factor = self.org_image.height() / float(height)
    image = image.scaled(width * scale_factor, height * scale_factor, Qt.IgnoreAspectRatio,
                         Qt.SmoothTransformation)
    return image


class MyMainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.model = create_model(False)
        self.isvideo = False
        self.is_pause = False

    def Open(self):
        sc_name, filetype = QFileDialog.getOpenFileName(caption="选取文件", directory=os.getcwd(),
                                                        filter="All Files (*)")
        if sc_name.split(".")[-1] != 'jpg' and sc_name.split(".")[-1] != 'png' and sc_name.split(".")[-1] != 'mp4':
            QMessageBox.critical(self, 'File Type Not Right', 'The type of file selected is not supported!',
                                 buttons=QMessageBox.Cancel)
        elif sc_name.split(".")[-1] == 'mp4':
            self.isvideo = True
            i = 0
            remove('./data/testA')
            capture = cv.VideoCapture(sc_name)
            self.frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
            self.frameRate = capture.get(cv.CAP_PROP_FPS)
            if capture.isOpened():
                while True:
                    ret, img = capture.read()
                    self.progressBar.setValue(i / self.frame_count * 100)
                    if not ret:
                        break
                    last_img = img
                    cv.imwrite('./data/testA/'+str(i)+'.jpg', img)
                    i = i + 1
            image = getImage(last_img, self)
            self.org_image.setPixmap(QPixmap.fromImage(image))
        else:
            img = cv.imread(sc_name)

            remove('./data/testA')
            cv.imwrite('./data/testA/1.jpg', img)

            image = getImage(img, self)
            self.org_image.setPixmap(QPixmap.fromImage(image))

    def Transfer(self):
        dataset = create_dataset('test')  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)
        self.model.setup()
        web_dir = os.path.join(opt['results_dir'], opt['name'],
                               '{}_{}'.format(opt['phase'], opt['epoch']))  # define the website directory
        webpage = html.HTML(web_dir,
                            'Experiment = %s, Phase = %s, Epoch = %s' % (opt['name'], opt['phase'], opt['epoch']))
        for i, data in enumerate(dataset):
            self.progressBar.setValue(i/dataset_size*100)
            if i >= opt['num_test']:  # only apply our model to opt.num_test images.
                break
            self.model.set_input(data)  # unpack data from data loader
            self.model.test()  # run inference
            visuals = self.model.get_current_visuals()  # get image results
            img_path = self.model.get_image_paths()  # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt['aspect_ratio'], width=opt['display_winsize'])
        webpage.save()  # save the HTML
        if not self.isvideo:
            self.progressBar.setValue(100)
            rst = cv.imread('./results/Chinese Painting Style/test_latest/images/1_fake_B.png')
            image = getImage(rst, self)
            self.after_image.setPixmap(QPixmap.fromImage(image))
            org = cv.imread('./results/Chinese Painting Style/test_latest/images/1_real_A.png')
            image = getImage(org, self)
            self.org_image.setPixmap(QPixmap.fromImage(image))
        else:
            self.progressBar.setValue(0)
            th = threading.Thread(target=self.Display)
            th.setDaemon(True)
            th.start()

    # display the video on the QLabel
    def Display(self):
        i = 0
        while i < self.frame_count:
            # calculate the processing time
            time_start = time.time()
            if self.is_pause:
                continue
            frame = cv.imread('./results/Chinese Painting Style/test_latest/images/'+str(i)+'_fake_B.png')
            org = cv.imread('./results/Chinese Painting Style/test_latest/images/'+str(i)+'_real_A.png')
            image = getImage(frame, self)
            self.after_image.setPixmap(QPixmap.fromImage(image))
            image = getImage(org, self)
            self.org_image.setPixmap(QPixmap.fromImage(image))
            time_end = time.time()
            time_wait = int(1000 / self.frameRate - 1000 * (time_end - time_start))
            if time_wait <= 0:
                time_wait = 1
            cv.waitKey(time_wait)
            self.progressBar.setValue(i/self.frame_count*100)
            i = i + 1

    def Pause(self):
        self.is_pause = not self.is_pause

    def Play(self):
        self.is_pause = False


if __name__ == "__main__":
    # 固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    app.setStyleSheet(StyleSheet)

    # 初始化
    myWin = MyMainForm()
    # 将窗口控件显示在屏幕上
    myWin.show()
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
