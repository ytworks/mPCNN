#! /usr/bin/env python
# coding:utf-8

import numpy as np
from scipy import signal


class mPCNN(object):


    def __init__(self, feature_map, weights, W,
                 ah = 0.1, vh = 2, at = 0.1, vt = 2):
        picture_size = feature_map.shape
        self.H = np.zeros(picture_size)
        self.Y = np.zeros((picture_size[0], picture_size[1]))
        self.U = np.zeros((picture_size[0], picture_size[1]))
        self.U_prev = np.zeros((picture_size[0], picture_size[1]))
        self.T = np.zeros((picture_size[0], picture_size[1]))
        self.S = feature_map
        self.ah, self.vh, self.at, self.vt = ah, vh, at, vt
        self.W = W
        self.weights = weights
        for i in range(40):
            self.train()


    def train(self):
        conv = signal.convolve2d(self.Y, self.W, 'same')
        stacked_conv = np.stack([conv for i in range(self.S.shape[2])], axis=-1)
        self.H = self.H * np.exp(-self.ah) + self.S + self.vh * stacked_conv
        U_temp = np.ones((self.S.shape[0], self.S.shape[1]))
        for k in range(self.S.shape[2]):
            U_temp *= 1.0 + np.dot(self.H, self.weights)
        self.U_prev = self.U
        self.U = U_temp
        for i in range(len(self.S)):
            for j in range(len(self.S[0])):
                self.Y[i][j] = 1.0 if self.U[i][j] > self.Y[i][j] else 0.0
        self.T = np.exp(-self.at) * self.T + self.vt * self.Y
        print np.mean(self.H), np.mean(self.U), np.mean(self.T)


    def get_image(self):
        return np.array((self.U - np.min(self.U)) / (np.max(self.U) - np.min(self.U)) * 255.0).astype(np.int32)

if __name__ == '__main__':
    import cv2
    p1 = cv2.imread('JPCLN114_2_roi_cam0.png')
    p2 = cv2.imread('JPCLN114_2_roi_cam_img0.png')
    feature_map = np.concatenate(((p1 -127.0)/ 127.0, (p2 -127)/ 127.0), axis = 2)
    weights = np.array([0.1/3, 0.1/3, 0.1/3, 0.9/3, 0.9/3, 0.9/3])
    W = np.array([[1.0 / np.sqrt(2), 1.0, 1.0 / np.sqrt(2)],
         [1, 0.0, 1],
         [1.0 / np.sqrt(2), 1, 1.0 / np.sqrt(2)]])
    obj = mPCNN(feature_map = feature_map,
                weights = weights,
                W = W)
    p3 = obj.get_image()
    thresh = int(np.mean(p3))
    max_pixel = 255
    blur = cv2.GaussianBlur(p3.astype(np.uint8),(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite('sample.png', th3)
