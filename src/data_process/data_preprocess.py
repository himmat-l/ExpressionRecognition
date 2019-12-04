import os
import cv2
import numpy as np
import keras
import sys
sys.path.append('../..')


class DataPreprocess(object):
	"""This class contains functions of preprocessing data, you can
	change the parameters in __init__ according to your needs
	"""

	def __init__(self):
		self.image_size = (224, 224)
		self.horizontal_flip = True
		self.rotation_range = 20
		self.zoom_range = 0.1
		self.shear_range = 0.1
		self.color_mode = 'grayscale'
		self.class_mode = 'categorical'

		super(DataPreprocess, self).__init__()

	def put_images_on_grid(self, images, shape=(8, 8), img_channels=3):
		"""Merge all images into a big one
		:param images: images to be merged
		:param shape: the image shape after merged, the default (8, 8) means at most 64 images are needed
		:param img_channels: 1 for gray scale, 3 for RGB
		:return: image after merged
		"""
		nrof_images = images.shape[0]
		img_size = images.shape[1]
		bw = 3
		img = np.zeros((shape[1] * (img_size + bw) + bw, shape[0] * (img_size + bw) + bw, img_channels), np.float32)
		for i in range(shape[1]):
			x_start = i * (img_size + bw) + bw
			for j in range(shape[0]):
				img_index = i * shape[0] + j
				if img_index >= nrof_images:
					break
				y_start = j * (img_size + bw) + bw
				img[x_start:x_start + img_size, y_start:y_start + img_size, :] = images[img_index, :, :, :]
			if img_index >= nrof_images:
				break
		return img

	def image_preprocess(self, face_img):
		"""
		 Convert RGB to GRAY;
		 Resize to 48 * 48;
		 Do equalizeHist;
		 scale to [0.0, 1.0].
		:param face_img: input image
		:return: image after transformed
		"""
		# face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
		face_img = cv2.resize(face_img, (224, 224))
		# face_img = cv2.equalizeHist(face_img)
		# face_img = face_img * (1.0 / 255.0)
		return face_img

	def image_preprocess_multi_patch(self, face_img):
		"""
		Convert RGB to GRAY
		Resize to 48 * 48
		Scale to [0.0,1.0]
		Crop image on the four corners and center;
		:param face_img:
		:return: Cropped images
		"""
		face_imgs = []
		face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
		face_img = face_img * (1.0 / 255)
		face_imgs.append(cv2.resize(face_img, (48, 48)))
		face_imgs.append(cv2.resize(face_img[0:46, 0:46], (48, 48)))
		face_imgs.append(cv2.resize(face_img[0:46, 2:48], (48, 48)))
		face_imgs.append(cv2.resize(face_img[2:48, 0:46], (48, 48)))
		face_imgs.append(cv2.resize(face_img[2:48, 2:48], (48, 48)))
		return face_imgs

	def train_data_angumentation(self, image):
		"""
		Data augumentation function for training data, used as an input for ImageDataGenerator in keras
		Including: Random Cropping; Resize; EqualizeHist; Rescale; Gaussian noise
		:param image: image to be transformed
		:return: image after transform
		"""
		# random_crop
		h, w = image.shape[:2]
		x1 = 0
		x2 = w
		y1 = 0
		y2 = h
		random_seed = np.random.randint(0, 4)
		if random_seed == 0:
			x1 = 4
		if random_seed == 1:
			x2 = w - 4
		if random_seed == 2:
			y1 = 4
		if random_seed == 3:
			y2 = h - 4
		image_crop = image[y1:y2, x1:x2, :]
		image_crop = cv2.resize(image_crop, image.shape[:2], interpolation=cv2.INTER_CUBIC)
		image_crop = np.array(image_crop, np.uint8)
		# hist equation
		image_crop = cv2.equalizeHist(image_crop)
		image_crop = np.array(image_crop, np.float64).reshape(image.shape)
		image_crop = image_crop / 255.0
		noise = np.random.normal(0, 0.01, image.shape)
		image_crop += noise
		image_crop = np.clip(image_crop, 0.0, 1.0)
		return image_crop

	def show_augmented_images(self):
		"""
		Show angumented images
		:return: None
		"""
		train_gen, val_gen, _ = self.get_data_generator_mixed_data_val('E:/expression_data/after_filter', batch_size=64)
		while True:
			train_imgs, _ = next(train_gen)
			img_grids = self.put_images_on_grid(train_imgs)
			img_grids = np.array(img_grids * 255.0, dtype=np.uint8)
			img_grids = np.squeeze(img_grids, axis=-1)
			print(img_grids.shape)
			cv2.imshow('pic', img_grids)
			key = cv2.waitKey(0)
			if key == ord('q'):
				break
			else:
				continue

	def test_data_angumentation(self, image):
		"""
		Transform test data, used as an input for ImageDataGenerator in keras
		:param image: test image
		:return: transformed test image
		"""
		# hist equation
		# image1 = image.astype(np.uint8)
		# image1 = cv2.equalizeHist(image1)
		# image1 = np.array(image1, np.float64).reshape(image.shape)
		image1 = image / 255.0
		return image1

	def get_data_generator(self, base_dir, batch_size=1):
		"""
		Get train, val, test data generator based on directory mode.
		You can modify the data angumentation mode in the function according to your needs.
		(refer to keras: https://keras.io/zh/preprocessing/image/)
		:param base_dir: base path for data(base_dir/train, base_dir/val, base_dir/test
						 is path for train, val, test data respectively)
		:param batch_size: batch size
		:return: generator for train, val, test data
		"""
		train_datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=self.rotation_range,
		                                                             horizontal_flip=self.horizontal_flip,
		                                                             zoom_range=self.zoom_range,
		                                                             shear_range=self.shear_range,
		                                                             preprocessing_function=self.train_data_angumentation)
		train_gen = train_datagen.flow_from_directory(os.path.join(base_dir, 'train'),
		                                              target_size=self.image_size,
		                                              color_mode=self.color_mode,
		                                              batch_size=batch_size,
		                                              class_mode=self.class_mode)

		val_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=self.test_data_angumentation)
		val_gen = val_datagen.flow_from_directory(os.path.join(base_dir, 'val'),
		                                          target_size=self.image_size,
		                                          color_mode=self.color_mode,
		                                          batch_size=batch_size,
		                                          class_mode=self.class_mode)
		return train_gen, val_gen, val_gen

	def get_data_generator_mixed_data(self, base_dir, batch_size=1):
		"""
		Not used any more
		:param base_dir:
		:param batch_size:
		:return:
		"""
		train_datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=self.rotation_range,
		                                                             horizontal_flip=self.horizontal_flip,
		                                                             zoom_range=self.zoom_range,
		                                                             shear_range=self.shear_range,
		                                                             preprocessing_function=self.train_data_angumentation)
		train_gen = train_datagen.flow_from_directory(os.path.join(base_dir, 'train'),
		                                              target_size=self.image_size,
		                                              color_mode=self.color_mode,
		                                              batch_size=batch_size,
		                                              class_mode=self.class_mode)

		val_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=self.test_data_angumentation)
		val_gen = val_datagen.flow_from_directory(os.path.join('../../data/CK+/SCK+', 'val'),
		                                          target_size=self.image_size,
		                                          color_mode=self.color_mode,
		                                          batch_size=batch_size,
		                                          class_mode=self.class_mode)
		return train_gen, val_gen, val_gen

	def get_data_generator_mixed_data_val(self, base_dir, batch_size=1):
		"""
		Split data into trian and val (0.9:0.1)
		:param base_dir: base path for data(base_dir/train is path for data)
		:param batch_size: batch size
		:return: generator for train, val, test data(test data is not used)
		"""
		train_datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=self.rotation_range,
		                                                             horizontal_flip=self.horizontal_flip,
		                                                             zoom_range=self.zoom_range,
		                                                             shear_range=self.shear_range,
		                                                             preprocessing_function=self.train_data_angumentation,
		                                                             validation_split=0.1)
		train_gen = train_datagen.flow_from_directory(os.path.join(base_dir, 'train'),
		                                              target_size=self.image_size,
		                                              color_mode=self.color_mode,
		                                              batch_size=batch_size,
		                                              class_mode=self.class_mode,
		                                              seed=2019,
		                                              subset='training')

		val_gen = train_datagen.flow_from_directory(os.path.join(base_dir, 'train'),
		                                            target_size=self.image_size,
		                                            color_mode=self.color_mode,
		                                            batch_size=batch_size,
		                                            class_mode=self.class_mode,
		                                            seed=2019,
		                                            subset='validation')
		return train_gen, val_gen, val_gen


if __name__ == '__main__':
	data_module = DataPreprocess()
	data_module.show_augmented_images()
