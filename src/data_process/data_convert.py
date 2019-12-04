#Convert datatype to tfrecord

import sys
sys.path.append('../../')
import os
import random
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
from keras.utils import to_categorical
from tqdm import tqdm
from mtcnn.mtcnn_detector import get_mtcnn_detector


class DataStyleConverter(object):
	def __init__(self, batch_size=32):
		super(DataStyleConverter, self).__init__()
		self.batch_size = batch_size
		self.ck_base_dir = '../../data'
		self.detector = get_mtcnn_detector(prefix=['../../mtcnn/MTCNN_model/PNet_landmark/PNet',
		                                           '../../mtcnn/MTCNN_model/RNet_landmark/RNet',
		                                           '../../mtcnn/MTCNN_model/ONet_landmark/ONet'])

	def int64_feature(self, values):
		"""Returns a TF-Feature of int64s.
		"""
		if not isinstance(values, (tuple, list)):
			values = [values]
		return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

	def float_feature(self, value):
		"""Wrapper for inserting float features into Example proto.
		"""
		if not isinstance(value, list):
			value = [value]
		return tf.train.Feature(float_list=tf.train.FloatList(value=value))

	def bytes_feature(self, value):
		"""Wrapper for inserting bytes features into Example proto.
		"""
		if not isinstance(value, list):
			value = [value]
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

	def ck_convert_to_tfrecord(self, filename, img_shape):
		images_dir = os.path.join(self.ck_base_dir, 'ck+/extended-cohn-kanade-images/cohn-kanade-images')
		labels_dir = os.path.join(self.ck_base_dir, 'ck+/Emotion_labels/Emotion')

		writer = tf.python_io.TFRecordWriter(filename)

		subjects_list = os.listdir(labels_dir)
		bar = tqdm(subjects_list[100:])
		for dex, sub in enumerate(bar):

			bar.set_description('当前图片处理进度: 第{}组'.format(dex))
			seq_dir = os.path.join(labels_dir, sub)
			seq_list = os.listdir(seq_dir)
			for seq in seq_list:
				label_dir = os.path.join(seq_dir, seq)
				label_list = os.listdir(label_dir)
				for label in label_list:
					lab = int(np.loadtxt(os.path.join(label_dir, label), dtype=int))
					img_dir = os.path.join(images_dir, sub, seq)
					images = os.listdir(img_dir)
					for image in images[:2]:
						print(os.path.join(img_dir, image))
						img0 = cv2.imread(os.path.join(img_dir, image), 0)
						img_rgb = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
						# 一张图片可能包含多个人脸
						faces, _ = self.detector.detect(img_rgb)
						try:
							for face in faces:
								x1, y1, x2, y2 = face[:4]
								x1 = max(0, int(x1) - 15)
								y1 = max(0, int(y1) - 15)
								x2 = min(img0.shape[1], int(x2) + 15)
								y2 = min(img0.shape[0], int(y2) + 15)
								face_img = img_rgb[y1:y2, x1:x2]
								face_img = cv2.resize(face_img, img_shape)
								img_raw = Image.fromarray(np.uint8(face_img)).tobytes()
								train_example0 = tf.train.Example(features=tf.train.Features(feature={
									'label': self.int64_feature(8),
									'image_raw': self.bytes_feature(img_raw)
								}))
								writer.write(train_example0.SerializeToString())
						except:
							continue
					for image in images[-5:]:
						img1 = cv2.imread(os.path.join(img_dir, image), 0)
						img_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
						faces, _ = self.detector.detect(img_rgb)
						try:
							for face in faces:
								x1, y1, x2, y2 = face[:4]
								x1 = max(0, int(x1) - 15)
								y1 = max(0, int(y1) - 15)
								x2 = min(img1.shape[1], int(x2) + 15)
								y2 = min(img1.shape[0], int(y2) + 15)
								face_img = img_rgb[y1:y2, x1:x2]
								face_img = cv2.resize(face_img, img_shape)
								img_raw = Image.fromarray(np.uint8(face_img)).tobytes()
								example = tf.train.Example(features=tf.train.Features(feature={
									'label': self.int64_feature(lab),
									'image_raw': self.bytes_feature(img_raw)
								}))
								writer.write(example.SerializeToString())
						except:
							continue
		writer.close()

	def read_tfrecords(self, filename):
		def pares_tf(example_proto):
			# 定义解析的字典
			dics = {
				'label': tf.FixedLenFeature([], tf.int64),
				'image_raw': tf.FixedLenFeature([], tf.string)}
			# 调用接口解析一行样本
			parsed_example = tf.parse_single_example(serialized=example_proto, features=dics)
			image = tf.decode_raw(parsed_example['image_raw'], out_type=tf.uint8)
			image = tf.reshape(image, shape=[224, 224, 3])
			# 这里对图像数据做归一化
			#image = (tf.cast(image, tf.float32) / 255.0)
			img_flip_h = tf.image.random_flip_left_right(image)
			random_satu = tf.image.random_saturation(img_flip_h, lower=0.2, upper=1.8)

			label = parsed_example['label']
			label = tf.reshape(label, shape=[1])
			label = tf.cast(label, tf.float32) - 1.0
			return random_satu, label

		dataset = tf.data.TFRecordDataset(filenames=[filename])
		dataset = dataset.map(pares_tf)
		dataset = dataset.shuffle(buffer_size=self.batch_size)
		dataset = dataset.batch(self.batch_size)
		dataset = dataset.prefetch(self.batch_size)
		dataset = dataset.make_initializable_iterator()
		next_element = dataset.get_next()

		return dataset, next_element


if __name__ == '__main__':
	# data_convert = DataStyleConverter()
	# data_convert.ck_convert_to_tfrecord(r'D:\SCUT\V5_Lab\CV\Codes\ExpressionRecognition-new\data\ck+\tfrecords\val.tfrecords', (224, 224))
	with tf.Session() as sess:
		data_convert = DataStyleConverter()
		dataset, next_element = data_convert.read_tfrecords(
			r'D:\SCUT\V5_Lab\CV\Codes\ExpressionRecognition-new\data\ck+\tfrecords\val.tfrecords')
		sess.run(tf.global_variables_initializer())
		sess.run(dataset.initializer)
		img, label = sess.run(next_element)
		label = to_categorical(label)
		img = Image.fromarray(img[15])
		img.show()
		print(label[15])






