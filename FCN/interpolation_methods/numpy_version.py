# -*- coding: utf-8 -*-
# @Time    : 9/11/20 12:11 PM
# @Author  : Zeqi@@
# @FileName: numpy_version.py
# @Software: PyCharm

import cv2
import time
import  numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt

class Interpolation:
    """
        也称零阶插值。它输出的像素灰度值就等于距离它映射到的位置最近的输入像素的灰度值。但当图像中包含像素之间灰度级有变化的细微结构时,最邻近算法会在图像中产生人为加工的痕迹。
        具体计算方法：对于一个目的坐标，设为 M(x,y)，通过向后映射法得到其在原始图像的对应的浮点坐标，设为 m(i+u,j+v)，其中 i,j 为正整数，u,v 为大于零小于1的小数(下同)，则待求象素灰度的值 f(m)。利用浮点 m 相邻的四个像素求f(m)的值。
    """
    def __init__(self):
        pass

    def nearest(self, img, new_size):
        """
        Nearest Neighbour interpolate for RGB  image

        :param image: rgb image
        :param target_size: tuple = (height, width)
        :return: None
        """
        new_w, new_h = new_size
        # height and width of the input img
        h, w = img.shape[0], img.shape[1]
        # new image with rgb channel
        new_img = np.zeros(shape=(new_h, new_w, 3), dtype='uint8')
        # scale factor
        s_h, s_c = (h * 1.0) / new_h, (w * 1.0) / new_w

        # insert pixel to the new img
        for i in range(new_h):
            for j in range(new_w):
                p_x = int(j * s_c)
                p_y = int(i * s_h)

                new_img[i, j] = img[p_y, p_x]

        return new_img

    def bilinear(self, img, new_size):
        """
        Bilinear interpolate for RGB  image

        new_image pixel: x, y
        original image matched pixel: x/scale, y/scale -> x_, y_
        Find its surround pixels in the original image: I(x_, y_), I(x_+1, y_), I(x_, y_+1), I(x_+1, y_+1)
        Decide the weights of its surround pixels by distance: d = x_ - np.floor(x_)
        out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

        :param image: rgb image
        :param target_size: tuple = (height, width)
        :return: None
        """
        new_w, new_h = new_size
        # height and width of the input img
        h, w = img.shape[0], img.shape[1]
        # new image with rgb channel
        new_img = np.zeros(shape=(new_h, new_w, 3), dtype='uint8')
        # scale factor
        s_h, s_c = (h * 1.0) / new_h, (w * 1.0) / new_w

        # insert pixel to the new img
        for y in range(new_h):
            for x in range(new_w):
                x_ = x * s_c
                y_ = y * s_h
                dx = x_ - np.floor(x_)
                dy = y_ - np.floor(y_)
                ix = int(x_)
                iy = int(y_)

                if ix >= (w-1):
                    ix = ix - 1

                if iy >= (h-1):
                    iy = iy - 1

                # new_img[y, x] = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

                A = np.array([[(1-dy), dy]])
                B = np.array([[img[iy, ix], img[iy, ix + 1]],
                              [img[iy + 1, ix], img[iy + 1, ix + 1]]])
                C = np.array([[(1-dx), dx]])
                C = np.transpose(C)
                for c in range(3):
                    new_img[y, x, c] = np.dot(np.dot(A,B[:, :, c]), C)

        print(np.min(new_img), np.max(new_img))

        return new_img

    def plot_(self, new_h, new_w, s_c, s_h):
        w, h = int(new_w*s_c), int(new_h*s_h)
        for y in range(new_h):
            for x in range(new_w):
                x_ = x * s_c
                y_ = y * s_h
                dx = x_ - np.round(x_)
                dy = y_ - np.round(y_)
                ix = int(np.round(x_))
                iy = int(np.round(y_))
                if y_ < new_h and x < new_w:
                    _img = np.ones(shape=(h, w, 3), dtype='uint8') * 255
                    plt.grid()
                    plt.imshow(_img)
                    plt.scatter(y_, x_, s=50, c='green')
                    plt.scatter(iy, ix, s=50, c='red')
                    plt.pause(0.00001)
                    plt.clf()

    def S(self, x):
        x = np.abs(x)
        if 0 <= x < 1:
            return 1 - 2 * x * x + x * x * x
        if 1 <= x < 2:
            return 4 - 8 * x + 5 * x * x - x * x * x
        else:
            return 0

    def bicubic(self, img, new_size):
        """
        Bilinear interpolate for RGB  image

        new_image pixel: x, y
        original image matched pixel: x/scale, y/scale -> x_, y_
        Find its surround pixels in the original image: I(x_, y_), I(x_+1, y_), I(x_, y_+1), I(x_+1, y_+1)
        Decide the weights of its surround pixels by distance: d = x_ - np.floor(x_)
        out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

        :param image: rgb image
        :param target_size: tuple = (height, width)
        :return: None
        """
        new_w, new_h = new_size
        # height and width of the input img
        h, w = img.shape[0], img.shape[1]
        # new image with rgb channel
        new_img = np.zeros(shape=(new_h, new_w, 3), dtype='uint8')
        # scale factor
        s_h, s_c = (h * 1.0) / new_h, (w * 1.0) / new_w

        # self.plot_(100, 100, s_c, s_h)

        # insert pixel to the new img
        for y in range(new_h):
            for x in range(new_w):
                x_ = x * s_c
                y_ = y * s_h
                dx = x_ - np.round(x_)
                dy = y_ - np.round(y_)
                ix = int(np.round(x_))
                iy = int(np.round(y_))


                if ix >= (w-3):
                    ix = ix - 3

                if iy >= (h-3):
                    iy = iy - 3


                A = np.array([[self.S(1+dy), self.S(dy), self.S(1-dy), self.S(2-dy)]])

                #A = np.array([[self.S(2-dy), self.S(1-dy), self.S(dy), self.S(1+dy)]])

                B = np.array([[img[iy-1, ix-1], img[iy-1, ix], img[iy-1, ix+1], img[iy-1, ix+2]],
                              [img[iy, ix-1], img[iy, ix], img[iy, ix+1], img[iy, ix+2]],
                              [img[iy+1, ix-1], img[iy+1, ix], img[iy+1, ix+1], img[iy+1, ix+2]],
                              [img[iy+2, ix-1], img[iy+2, ix], img[iy+2, ix+1], img[iy+2, ix+2]]])

                C = np.array([[self.S(1 + dx), self.S(dx), self.S(1 - dx), self.S(2 - dx)]])
                #C = np.array([[self.S(2 - dx), self.S(1 - dx), self.S(dx), self.S(1 + dx)]])
                C = np.transpose(C)
                for c in range(3):
                    pixel = np.dot(np.dot(A, B[:, :, c]), C)

                    new_img[y, x, c] = pixel

                # print(np.shape(new_img))
        print(np.min(new_img), np.max(new_img))

        return new_img


def Timer(time_1, time_2):
    return np.round((time_2-time_1)*1000, 2)

def plot_show(original, opencv, mine, difference):
    fig = plt.figure()
    images_BGR = [original, opencv, difference, mine]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_BGR]
    names = ['Original', 'Opencv', 'Difference','Numpy']
    for i, title in enumerate(names):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i])
        plt.title(title)
        plt.axis('off')
    plt.show()

if __name__=='__main__':
    image = cv2.imread('test.jpg')
    image = cv2.resize(image, (52, 24), interpolation=cv2.INTER_NEAREST)
    print('Input size: ', np.shape(image))
    s_time = time.time()
    nearest_cv = cv2.resize(image, (520, 240), interpolation=cv2.INTER_NEAREST)
    c_time = time.time()
    print('Opencv nearest interpolation: {0} ms'.format(Timer(s_time, c_time)))
    nearest_img = Interpolation().nearest(image, (520, 240))
    e_time = time.time()
    print('numpy nearest interpolation: {0} ms'.format(Timer(c_time, e_time)))
    print('Whether they are equivalent? ', np.all((nearest_cv - nearest_img) == 0))

    # plot_show(image, nearest_cv, nearest_img, nearest_cv-nearest_img)


    # # Bilinear interpolation
    # s_time = time.time()
    # linear_cv = cv2.resize(image, (520, 240), interpolation=cv2.INTER_LINEAR)
    # c_time = time.time()
    # print('Opencv Bilinear interpolation: {0} ms'.format(Timer(s_time, c_time)))
    # bilinear_img = Interpolation().bilinear(image, (520, 240))
    # e_time = time.time()
    # print('numpy Bilinear interpolation: {0} ms'.format(Timer(c_time, e_time)))
    # print('Whether they are equivalent? ', np.all((linear_cv - bilinear_img) == 0))
    # plot_show(image, linear_cv, bilinear_img, linear_cv - bilinear_img)



    #  Bicubic interpolation
    s_time = time.time()
    bicubic_cv = cv2.resize(image, (520, 240), interpolation=cv2.INTER_CUBIC)
    c_time = time.time()
    print('Opencv Bicubic interpolation: {0} ms'.format(Timer(s_time, c_time)))
    bicubic_img = Interpolation().bicubic(image, (520, 240))
    e_time = time.time()
    print('numpy bicubic interpolation: {0} ms'.format(Timer(c_time, e_time)))
    print('Whether they are equivalent? ', np.all((bicubic_cv - bicubic_img) == 0))
    plot_show(image, bicubic_cv, bicubic_img, bicubic_cv - bicubic_img)


