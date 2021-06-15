import cv2 as cv
import os
import numpy as np
import matplotlib.image as mp
from skimage import img_as_ubyte
from PIL import Image


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]


def extract_edge(frame):
    # Load the model.
    net = cv.dnn.readNet(cv.samples.findFile('deploy.prototxt'), cv.samples.findFile('hed_pretrained_bsds.caffemodel'))
    inp = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(frame.shape[1], frame.shape[0]),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    edge = net.forward()
    edge = edge[0, 0]
    edge = cv.resize(edge, (frame.shape[1], frame.shape[0]))
    edge = cv.merge([edge, edge, edge])
    frame = frame/2 + frame/2 * edge
    return frame


if __name__ == '__main__':
    # ! [Register]
    cv.dnn_registerLayer('Crop', CropLayer)
    # ! [Register]
    path = "..\\data\\org_trainA\\"  # 图像读取地址
    savepath = "..\\data\\trainA\\"  # 图像保存地址
    filelist = os.listdir(path)  # 打开对应的文件夹

    for item in filelist:
        name = path + item
        img = cv.imread(name)
        rst = extract_edge(img)
        save_name = savepath + item
        print(item+' is processed and saved to '+save_name)
        cv.imwrite(save_name, rst)
