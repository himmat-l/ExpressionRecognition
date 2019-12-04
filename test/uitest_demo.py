import sys
sys.path.append('../..')
import numpy as np
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
emotion_path = './emojis'
labels = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 创建屏幕窗口
window = tk.Tk()
window.title('表情识别软件')
window.geometry('960x1080')
window.iconbitmap("favicon.ico")
window["background"] = "blue"
# window.iconbitmap('xx.ico')
cap = cv2.VideoCapture(0)
detect_img = None
detect_reimg = None
camera_img = None
rec_img = None
pre = None
emoji = None
class UiTest(object):
	def __init__(self):
		self.detector = get_mtcnn_detector()
		self.data_module = DataPreprocess()
		self.sess = tf.Session()
		self.log_path = weights_path
		self.frame = None
		self.frames = 0
		self.x1 = 0
		self.y1 = 0
		self.x2 = 0
		self.y2 = 0
		self.rec_time = 0
		self.inputs = tf.placeholder(tf.float32,
		                             shape=(None, 224, 224, 3))
		self.logits, _ = MobileNet_v3().build_model_small(self.inputs, reduction_ratio=4, reuse=None)
		self.frame_left_up = tk.Frame(window, width=460, height=740)
		self.frame_left_up.place(x=0, y=0)
		self.frame_right_up = tk.Frame(window, width=500, height=390, bg='blue')
		self.frame_right_up.place(x=460, y=0)
		self.frame_left_down = tk.Frame(window, width=460, height=340, bg='pink')
		self.frame_left_down.place(x=0, y=740)
		self.frame_right_down = tk.Frame(window, width=500, height=690, bg='yellow')
		self.frame_right_down.place(x=460, y=390)
		super(UiTest, self).__init__()

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

	def open_camera(self):
		global camera_img
			# top = tk.Toplevel(window, width=460, height=290)
		while cap.isOpened():
			# print('reading')
			ret, self.frame = cap.read()
			self.frame = cv2.resize(self.frame, (480, 640))
			# print('frame shape', self.frame.shape)
			# print(frame.shape)
			if not ret:
				print('wrong')
				break

			canvas1 = tk.Canvas(self.frame_left_up, bg='gray', width=460, height=640)
			# img = cv2.resize(frame, (400, 600))
			cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGBA)
			camera_img = Image.fromarray(cv2image)
			camera_img = ImageTk.PhotoImage(camera_img)
			canvas1.create_image(0, 100, image=camera_img, anchor='nw')
			# cv2.waitKey(5)
			canvas1.place(x=0, y=100)
			self.frame_left_up.update_idletasks()

	def detect_face(self):
		global detect_img, detect_reimg
		print('detecting')
		start = time.time()
		face_imgs = []
		boxs = []
		# ret, self.frame = cap.read()
		canvas2 = tk.Canvas(self.frame_left_down, bg='gray', width=460, height=290)
		canvas2.place(x=0, y=50)
		img = cv2.resize(self.frame, (224, 224))
		cv2.imwrite(r'./save_img/img.jpg', self.frame)
		# cv2.imshow('img', img)
		# cv2.waitKey(0)
		# detect_img = cv2.imread(r'./save_img/img.jpg')
		detect_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
		detect_img = Image.fromarray(detect_img)
		detect_img = ImageTk.PhotoImage(detect_img)
		canvas2.create_image(0, 0, image=detect_img, anchor='nw')
		faces, landmarks = self.detector.detect(self.frame)
		if len(faces) != 0:
			for face in faces:
				x1, y1, x2, y2 = face[:4]
				self.x1 = max(0, int(x1) - 5)
				self.y1 = max(0, int(y1) - 5)
				self.x2 = min(640, int(x2) + 5)
				self.y2 = min(480, int(y2) + 5)
				if self.y2 <= self.y1 or self.x2 <= self.x1:
					continue
				reimg = self.frame[self.y1:self.y2, self.x1:self.x2, :]
				reimg = self.preprocess_image(reimg)
				cv2.imwrite(r'./save_img/face_img.jpg', reimg)
				detect_reimg = cv2.imread(r'./save_img/face_img.jpg')
				detect_reimg = cv2.cvtColor(detect_reimg, cv2.COLOR_BGR2RGBA)
				detect_reimg = Image.fromarray(detect_reimg)
				detect_reimg = ImageTk.PhotoImage(detect_reimg)
				canvas2.create_image(240, 0, image=detect_reimg, anchor='nw')
			# canvas2.place(x=0, y=50)
		txt_show = tk.Label(self.frame_left_down, text='检测用时：' + str('{:.2f}'.format(time.time() - start)+'s'), font=('Arial',15), bg='red')
		txt_show.place(x=0, y=0)
		# self.frame_left_down.update_idletasks()
		print('detecting end')
		# self.frame_left_down.update()

	def load_weights(self):
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
		weights = tf.train.get_checkpoint_state(self.log_path)
		saver.restore(self.sess, weights.model_checkpoint_path)

	def softmax(self, x):
		x_exp = np.exp(x)
		# 如果是列向量，则axis=0
		x_sum = np.sum(x_exp, axis=0, keepdims=True)
		s = x_exp / x_sum
		return s

	def rec_face(self):
		global rec_img
		global pre
		global emoji
		# face_imgs = []
		pres = []
		boxs = []
		start = time.time()
		self.load_weights()
		canvas3 = tk.Canvas(self.frame_right_down, width=480, height=640)
		canvas3.place(x=0, y=50)
		img = cv2.imread(r'./save_img/img.jpg')
		# img = cv2.resize(img, (480, 640))
		face_img = cv2.imread(r'./save_img/face_img.jpg')

		face_imgs = np.array(face_img).reshape((-1, 224, 224, 3))
		pres = self.sess.run(self.logits, feed_dict={self.inputs: face_imgs})
		self.rec_time = time.time() - start
		for p in pres:
			# print(p.shape)
			pre = self.softmax(p)
			print(pre)
			idx = np.argmax(pre)
			# rec_img = self.postprocess_iamge(img, pre, x1, y1, x2, y2)
			emoji = cv2.imread(os.path.join(emotion_path, labels[int(idx)] + '.png'))
			emoji = cv2.resize(emoji, (240, 240))
			mask = cv2.cvtColor(emoji, cv2.COLOR_RGB2GRAY)
			ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
			inv_mask = cv2.bitwise_not(mask)
			ori = img[self.y1:self.y1 + 240, self.x1:self.x1 + 240, :]
			# print(ori.shape)
			ori = cv2.bitwise_and(ori, ori, mask=inv_mask)
			emo = cv2.bitwise_and(emoji, emoji, mask=mask)
			emotions = emo + ori
			img[self.y1:self.y1 + 240, self.x1:self.x1 + 240, :] = emotions
			rec_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
			rec_img = Image.fromarray(rec_img)
			rec_img = ImageTk.PhotoImage(rec_img)
			canvas3.create_image(0, 0, image=rec_img, anchor='nw')

		txt_show = tk.Label(self.frame_right_down, text='表情类别：' + labels[int(idx)], font=('Arial',15), bg='red')
		txt_show.place(x=0, y=0)

	def show_details(self):
		time_show = tk.Label(self.frame_right_up, text='检测用时：' + str('{:.2f}'.format(self.rec_time)+'s'), font=('Arial', 15), bg='red')
		time_show.place(x=0, y=0)
		labels = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprise', 'neutral']
		for i in range(len(labels)):
			tk.Label(self.frame_right_up, text=labels[i]+': {:.2f}%'.format(pre[i]/np.sum(pre)),
			         font=('Arial', 15)).place(x=50, y=(55 if i==0 else 55+i*60))

	def clear_image(self):
		pass

	def thread_it(self,func, *args):
		t = td.Thread(target=func, args=args)
		t.setDaemon(True)
		t.start()

	def ui_test(self):
		frame_button = tk.Frame(window, width=460, height=100, relief='groove', borderwidth=0).place(x=0, y=0)
		open_cam_button = tk.Button(frame_button, text='open camera', font=('Arial', 12), command=lambda: self.thread_it(self.open_camera))
		open_cam_button.place(x=5, y=5)
		detect_button = tk.Button(frame_button, text='detect face', font=('Arial', 12), command=lambda: self.thread_it(self.detect_face))
		detect_button.place(x=115, y=5)
		pre_button = tk.Button(frame_button, text='recognize face', font=('Arial', 12), command=self.rec_face)
		pre_button.place(x=210, y=5)
		detail_button = tk.Button(frame_button, text='show details', font=('Arial', 12), command=self.show_details)
		detail_button.place(x=5, y=65)
		clear_button = tk.Button(frame_button, text='clear image', font=('Arial', 12), command=self.clear_image)
		clear_button.place(x=115, y=65)
		# detect_button = tk.Button(frame_button, text='detect face', font=('Arial', 10), command=self.detect_face)
		# detect_button.place(x=90, y=0)
		window.mainloop()

if __name__ == '__main__':
	ui_test = UiTest()
	ui_test.ui_test()