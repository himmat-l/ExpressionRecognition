import sys
sys.path.append('../..')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import time
import os
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk
import threading as td
from src.data_process.data_preprocess import DataPreprocess
from src.models.mobilenet_v3 import MobileNet_v3
from mtcnn.mtcnn_detector import get_mtcnn_detector



weights_path = '../models/ck+/'
# img_path = 'testdata/2.jpg'
emotion_path = './emojis'
labels = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# labels = ['neutral', 'happy', 'doubt', 'understand', 'sleepy', 'haqian', 'haqian_v2']

# frame = None
# frames = 0



class Test(object):
    """
    Demo for single frame based classification
    Attributes:
        detector: MTCNN for face detection
        classifier: expression classification model
    """
    def __init__(self, log_path=weights_path):  # log_path='../models/resnet/mixeddata/cm_loss_0.01'
        self.detector = get_mtcnn_detector()
        self.image_size = (224, 224)
        self.img_channels = 3
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.image_size[0], self.image_size[1], self.img_channels))
        self.logits, _ = MobileNet_v3().build_model_small(self.inputs, reduction_ratio=4, reuse=None)
        self.log_path = log_path
        self.data_module = DataPreprocess()
        self.sess = tf.Session()
        self.frame = None
        self.frames = 0

    def load_weights(self):
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        weights = tf.train.get_checkpoint_state(self.log_path)
        saver.restore(self.sess, weights.model_checkpoint_path)


    def preprocess_image(self, face_img):
        """
        Preprocess extracted face image before classification
        :param face_img: input face image
        :return: face image after preprocessing
        """
        face_img = self.data_module.image_preprocess(face_img)
        face_img = np.array(face_img)
        # face_img = np.expand_dims(face_img, axis=-1)
        # face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def postprocess_iamge(self, frame, pre, x1, y1, x2, y2):
        """
        Postprocessing for each frame after classification, including add emoijs, draw face boxes,
        draw probability hist
        :param frame: frame
        :param pre: predict result
        :param x1: face coordinate predicted by mtcnn
        :param y1: face coordinate predicted by mtcnn
        :param x2: face coordinate predicted by mtcnn
        :param y2: face coordinate predicted by mtcnn
        :return: frame
        """
        idx = np.argmax(pre)
        print(frame.shape)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frame, labels[int(idx)], (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
        emotions = cv2.imread(os.path.join(emotion_path, labels[int(idx)] + '.png'))
        emotions = cv2.resize(emotions, (40, 40))
        mask = cv2.cvtColor(emotions, cv2.COLOR_RGB2GRAY)
        ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)
        ori = frame[y1:y1 + 40, x1:x1 + 40, :]
        print(ori.shape)
        ori = cv2.bitwise_and(ori, ori, mask=inv_mask)
        emo = cv2.bitwise_and(emotions, emotions, mask=mask)
        emotions = emo + ori
        frame[y1:y1 + 40, x1:x1 + 40, :] = emotions
        return frame

    def single_test(self, img_path):
        """
        Test for a single image
        :param img: input image
        :return: None
        """
        plt.ion()
        start = time.time()
        self.load_weights()
        # self.sess.run(tf.global_variables_initializer())
        img = cv2.imread(img_path, 0)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # print('img_rgb_shape: ', img_rgb.shape)
        faces, landmarks = self.detector.detect(img_rgb)
        face_imgs = []
        pres = []
        boxs = []
        for face in faces:
            x1, y1, x2, y2 = face[:4]
            x1 = max(0, int(x1) - 15)
            y1 = max(0, int(y1) - 15)
            x2 = min(img.shape[1], int(x2) + 15)
            y2 = min(img.shape[0], int(y2) + 15)
            face_img = img_rgb[y1:y2, x1:x2]
            # print('face_img_shape: ', face_img.shape)
            face_img = self.preprocess_image(face_img)
            face_imgs.append(face_img)
            boxs.append([x1, y1, x2, y2])
        if len(face_imgs) != 0:
            face_imgs = np.array(face_imgs).reshape((-1, 224, 224, 3))
            pres = self.sess.run(self.logits, feed_dict={self.inputs:face_imgs})
            # print('pres:', pres)
        for pre, box in zip(pres, boxs):
            x1, y1, x2, y2 = box
            # sns.barplot(x=labels, y=pre)
            # plt.ylim(0, 1)
            img = self.postprocess_iamge(img_rgb, pre, x1, y1, x2, y2)
        end = time.time()
        # cv2.putText(img, 'time:%.2f' % (end - start), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 1)
        cv2.imshow('single_img', img)
        # plt.show()
        # plt.pause(0.001)
        # plt.clf()
        if cv2.waitKey(0) & 0XFF == ord('q'):
            return

    def camera_test(self, interval=1):
        """
        Camera based test
        :param interval: interval between predictions
        :return: None
        """
        plt.ion()
        self.load_weights()
        # video_path = 'E:/surveillance/videos/sunell_xuan_gua/0/1.avi'
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        frames = 0
        pres = []
        while True:
            frames += 1
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                print('wrong')
                return
            if frames % interval == 0:
                faces, landmarks = self.detector.detect(frame)
                face_imgs = []
                # pres.clear()
                boxs = []
                if len(faces) != 0:
                    for face in faces:
                        x1, y1, x2, y2 = face[:4]
                        x1 = max(0, int(x1) - 5)
                        y1 = max(0, int(y1) - 5)
                        x2 = min(480, int(x2) + 5)
                        y2 = min(640, int(y2) + 5)
                        if y2 <= y1 or x2 <= x1:
                            continue
                        face_img = frame[y1:y2, x1:x2, :]
                        face_img = self.preprocess_image(face_img)
                        face_imgs.append(face_img)
                        boxs.append([x1, y1, x2, y2])
                if len(face_imgs) != 0:
                    face_imgs = np.array(face_imgs).reshape((-1, 224, 224, 3))
                    pres = self.sess.run(self.logits, feed_dict={self.inputs: face_imgs})
                    # for pre, box in zip(pres['result'], boxs):
                    for pre, box in zip(pres, boxs):
                        x1, y1, x2, y2 = box
                        # start4 = time.time()
                        # sns.barplot(x=labels, y=pre)
                        # plt.ylim(0, 1)
                        # print('time4:', time.time()-start4)

                        frame = self.postprocess_iamge(frame, pre, x1, y1, x2, y2)

            else:
                if len(pres) != 0:
                    for pre, box in zip(pres, boxs):
                        pass
                        # sns.barplot(x=labels, y=pre)
                        # plt.ylim(0, 1)

            end = time.time()
            cv2.putText(frame, 'time:%.2f' % (end - start), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 1)
            cv2.imshow('frame', frame)
            plt.show()
            plt.pause(0.001)
            plt.clf()
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break












if __name__ == '__main__':
    ct = Test()
    ct.camera_test()
    # ct.single_test(r'./test_data/3.jpg')
    # ct.single_test(r'.\test_data\1.jpg')
    # img_path = 'testdata/2.jpg'
    # image = cv2.imread(img_path, 0)
    # image = np.array(image/255.0, dtype=float)
    # image = np.expand_dims(image, -1)
    # noise = np.random.normal(0, 0.05, image.shape)
    # image += noise
    # image = np.clip(image, 0.0, 1.0)
    # image = np.array(image * 255, dtype=np.uint8)
    # print(image.shape)
    # cv2.imshow('pic', image)
    # cv2.waitKey(0)
