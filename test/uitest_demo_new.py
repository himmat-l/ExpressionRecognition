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

window = tk.Tk()
window.title('表情识别软件')
window.geometry('1024x768')
window.iconbitmap("favicon.ico")
main_frame = tk.Frame(window, width=1024, height=768, bg='white')
main_frame.pack()
frame_left = tk.Frame(main_frame, width=640, height=768, bg='white')
frame_left.place(x=0, y=0, anchor='nw')
frame_right = tk.Frame(main_frame, width=380, height=768, bg='lightblue')
frame_right.place(x=644, y=0, anchor='nw')
cap = cv2.VideoCapture(0)
cap.set(5, 10)
print(cap.get(5))
# 全局变量设置
detect_img = None
detect_reimg = None
camera_img = None
rec_img = None
emoji = None
canvas3_img = None
reimgs = []
face_imgs = []
class UiTest(object):
	def __init__(self):
		self.detector = get_mtcnn_detector()
		self.data_module = DataPreprocess()
		self.sess = tf.Session()
		self.log_path = weights_path
		self.frame = None
		self.frames = 0
		self.x1 = []
		self.y1 = []
		self.x2 = []
		self.y2 = []
		self.label_list = []
		self.rec_time = 0
		self.canvas1 = tk.Canvas(frame_left, bg='gainsboro', width=640, height=480)
		self.canvas1.place(x=0, y=45)
		self.canvas2 = tk.Canvas(frame_left, bg='gainsboro', width=640, height=224)
		self.canvas2.place(x=0, y=540)
		self.canvas3 = tk.Canvas(frame_right, bg='gainsboro', width=380, height=768)
		self.canvas3.place(x=0, y=448)
		self.inputs = tf.placeholder(tf.float32,
		                             shape=(None, 224, 224, 3))
		self.logits, _ = MobileNet_v3().build_model_small(self.inputs, reduction_ratio=4, reuse=None)
		super(UiTest, self).__init__()

	def preprocess_image(self, face_img):
		"""
		Preprocess extracted face image before classification
		:param face_img: input face image
		:return: face image after preprocessing
		"""
		face_img = self.data_module.image_preprocess(face_img)
		face_img = np.array(face_img)
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
		while cap.isOpened():
			ret, self.frame = cap.read()
			if not ret:
				print('wrong')
				break
			self.frame = cv2.flip(self.frame, 1)
			img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGBA)
			img = Image.fromarray(img)
			img = ImageTk.PhotoImage(img)
			self.canvas1.create_image(0, 0, anchor='nw', image=img)

	def detect_face(self):
		global detect_img, detect_reimg
		global reimgs
		print('detecting')
		start = time.time()
		del face_imgs[:]
		img = cv2.resize(self.frame, (224, 224))
		cv2.imwrite(r'./save_img/img.jpg', self.frame)
		detect_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
		detect_img = Image.fromarray(detect_img)
		detect_img = ImageTk.PhotoImage(detect_img)
		self.canvas2.create_image(0, 0, image=detect_img, anchor='nw')
		faces, landmarks = self.detector.detect(self.frame)
		del self.x1[:]
		del self.x2[:]
		del self.y1[:]
		del self.y2[:]
		if len(faces) != 0:
			del reimgs[:]
			for i, face in enumerate(faces):
				x1, y1, x2, y2 = face[:4]
				x1 = max(0, int(x1) - 5)
				y1 = max(0, int(y1) - 5)
				x2 = min(640, int(x2) + 5)
				y2 = min(480, int(y2) + 5)
				self.x1.append(x1)
				self.x2.append(x2)
				self.y1.append(y1)
				self.y2.append(y2)
				if y2 <= y1 or x2 <= x1:
					continue
				reimg = self.frame[y1:y2, x1:x2, :]
				reimg = self.preprocess_image(reimg)
				face_imgs.append(reimg)
				cv2.imwrite(r'./save_img/face_img{}.jpg'.format(i), reimg)
				detect_reimg = cv2.cvtColor(reimg, cv2.COLOR_BGR2RGBA)
				detect_reimg = Image.fromarray(detect_reimg)
				detect_reimg = ImageTk.PhotoImage(detect_reimg)

				reimgs.append(detect_reimg)
			for i in range(len(reimgs)):
				self.canvas2.create_image(225*(i+1), 0, image=reimgs[i], anchor='nw')
		else:
			self.canvas2.create_text(320, 100, text="未检测到人脸", font=('Arial', 15), anchor='nw')
		txt_show = tk.Label(frame_left, text='检测用时：' + str('{:.2f}'.format(time.time() - start)+'s'), font=('Arial',15), bg='red')
		txt_show.place(x=0, y=528)
		frame_left.update_idletasks()
		print('detecting end')

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
		global emoji
		global face_imgs
		global canvas3_img
		start = time.time()
		self.load_weights()
		self.canvas3.delete(canvas3_img)
		img = cv2.imread(r'./save_img/img.jpg')
		face_imgs_array = np.array(face_imgs).reshape((-1, 224, 224, 3))
		self.pres = self.sess.run(self.logits, feed_dict={self.inputs: face_imgs_array})
		self.rec_time = time.time() - start
		for p, x1, x2, y1, y2 in zip(self.pres, self.x1, self.x2, self.y1, self.y2):
			pre = self.softmax(p)
			idx = np.argmax(pre)
			emoji = cv2.imread(os.path.join(emotion_path, labels[int(idx)] + '.png'))
			emoji = cv2.resize(emoji, (x2-x1, y2-y1))
			mask = cv2.cvtColor(emoji, cv2.COLOR_RGB2GRAY)
			ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
			inv_mask = cv2.bitwise_not(mask)
			ori = img[y1:y2, x1:x2, :]
			ori = cv2.bitwise_and(ori, ori, mask=inv_mask)
			emo = cv2.bitwise_and(emoji, emoji, mask=mask)
			emotions = emo + ori
			img[y1:y2, x1:x2, :] = emotions
			cv2.imwrite(r'./save_img/canvas3.jpg', img)
			rec_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
			rec_img = cv2.resize(rec_img, (320, 320))
			rec_img = Image.fromarray(rec_img)
			rec_img = ImageTk.PhotoImage(rec_img)
		canvas3_img = self.canvas3.create_image(25, 0, image=rec_img, anchor='nw', tags="pic")
		if len(self.pres)==1:
			txt_show = tk.Label(frame_right, text='表情类别：' + labels[int(np.argmax(self.pres[0]))], width=35, font=('Arial', 15), bg='red')
		else:
			txt_show = tk.Label(frame_right, text='表情类别：' + labels[int(np.argmax(self.pres[0]))] + '、' + labels[int(np.argmax(self.pres[1]))], width=35, font=('Arial',15), bg='red')
		txt_show.place(x=0, y=418)
		frame_right.update()

	def show_details(self):
		while len(self.label_list) != 0:
			label = self.label_list.pop()
			label.destroy()
		time_show = tk.Label(frame_right, text='检测用时：' + str('{:.2f}'.format(self.rec_time)+'s'), width=35, font=('Arial', 15), bg='red')
		time_show.place(x=0, y=0)
		labels = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprise', 'neutral']
		for i, p in enumerate(self.pres):
			p = self.softmax(p)
			for j in range(len(labels)):
				lab = tk.Label(frame_right, text=labels[j]+': {:.2f}%'.format((p[j]/np.sum(p))*100),
				         font=('Arial', 15))
				lab.place(x=5+180*i, y=(55 if j == 0 else 55+j*45))
				self.label_list.append(lab)
		frame_right.update_idletasks()

	def quit(self):
		window.destroy()

	def thread_it(self, func, *args):
		t = td.Thread(target=func, args=args)
		t.setDaemon(True)
		t.start()

	def ui_test(self):
		open_cam_button = tk.Button(frame_left, text='open camera', font=('Arial', 12), command=lambda: self.thread_it(self.open_camera))
		open_cam_button.place(x=5, y=7)
		detect_button = tk.Button(frame_left, text='detect face', font=('Arial', 12), command=lambda: self.thread_it(self.detect_face))
		detect_button.place(x=115, y=7)
		pre_button = tk.Button(frame_left, text='recognize face', font=('Arial', 12), command=self.rec_face)
		pre_button.place(x=210, y=7)
		detail_button = tk.Button(frame_left, text='show details', font=('Arial', 12), command=self.show_details)
		detail_button.place(x=330, y=7)
		exit_button = tk.Button(frame_left, text='exit', font=('Arial', 12), command=self.quit)
		exit_button.place(x=435, y=7)
		window.mainloop()


if __name__ == '__main__':
	ui_test = UiTest()
	ui_test.ui_test()

