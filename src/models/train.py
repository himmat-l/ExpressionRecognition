"""
Train model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../..')
import tensorflow as tf
import os
from src.data_process import data_convert
from models.mobilenet_v3 import MobileNet_v3
from keras.utils import to_categorical

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class TrainModel(object):
	def __init__(self, class_num=8, batch_size=32, base_path='../../data'):
		self.base_path = base_path
		self.data_path = os.path.join(self.base_path, 'ck+/tfrecords')
		self.label_path = os.path.join(self.base_path, 'ck+/tfrecords')
		self.class_num = class_num
		self.batch_size = batch_size
		self.image_size = (224, 224)
		self.img_channels = 3
		self.reduction_ratio = 4
		self.data_convert = data_convert.DataStyleConverter(batch_size=self.batch_size)
		self.build_model = MobileNet_v3(self.batch_size, self.class_num)
		self.log_path = '../../models/ck+'

	def train(self):
		'''
		Train model
		:return:
		'''

		with tf.Graph().as_default():
			global_step = tf.Variable(0, trainable=False)
			train_dataset, train_iterator = self.data_convert.read_tfrecords(os.path.join(self.data_path, 'train.tfrecords'))
			val_dataset, val_iterator = self.data_convert.read_tfrecords(os.path.join(self.data_path, 'val.tfrecords'))
			inputs = tf.placeholder(tf.float32, shape=(None, self.image_size[0], self.image_size[1], self.img_channels))
			outputs = tf.placeholder(tf.int32, shape=(None, self.class_num))

			y, end_points = self.build_model.build_model_small(inputs, self.reduction_ratio)
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=outputs, logits=y))
			acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1)), tf.float32))
			total_loss = loss + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			# 滑动均值
			ema = tf.train.ExponentialMovingAverage(0.999, global_step)
			ema_opt = ema.apply([loss, acc])
			lr = tf.train.exponential_decay(0.001, global_step, 320 * 5, 0.1, staircase=True)
			opt = tf.train.AdamOptimizer(lr)
			grads = opt.compute_gradients(total_loss, var_list=tf.trainable_variables())
			apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies([apply_gradient_op] + update_ops):
				train_op = tf.no_op(name='train')

			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			sess = tf.Session(config=config)
			saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
			# weights = tf.train.get_checkpoint_state(r'D:\SCUT\V5_Lab\CV\Codes\ExpressionRecognition-new\models\ck+')
			# saver.restore(sess, weights.model_checkpoint_path)
			sess.run(tf.global_variables_initializer())
			with sess.as_default():
				step = 0
				val_loss = 0
				val_acc = 0
				best_val_acc = 0
				max_epochs = 100
				epoch = 1
				sess.run(train_dataset.initializer)
				sess.run(val_dataset.initializer)
				print('start training!')
				while epoch < max_epochs:
					while True:
						try:
							step += 1
							train_images, train_labels = sess.run(train_iterator)
							train_labels = to_categorical(train_labels)
							_, _, loss_, loss_ave_, lr_, acc_, acc_ave_ = sess.run(
							[train_op, ema_opt, loss, ema.average(loss), lr, acc, ema.average(acc)],
							feed_dict={inputs: train_images, outputs: train_labels})
							print('\rEpoch:%d Step:%d loss_ave:%.3f acc_ave:%.3f val_loss:%.3f val_acc:%.3f  lr:%.5f' % (
							epoch, step, loss_ave_, acc_ave_, val_loss, val_acc, lr_), end='')
							sys.stdout.flush()  # 刷新缓冲区

						except:
							# sess.run(train_dataset.initializer)
							# sess.run(val_dataset.initializer)
							# if epoch % 5 == 0:
							# 	checkpoint_path = os.path.join(self.log_path, 'model.ckpt')
							# 	saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
							epoch += 1
							step = 0
							val_steps = 0
							val_loss = 0.0
							val_acc = 0.0
							while True:
								try:

									val_images, val_labels = sess.run(val_iterator)
									val_labels = to_categorical(val_labels)
									val_loss_, val_acc_ = sess.run(
										[loss, acc],
										feed_dict={inputs: val_images, outputs: val_labels}
									)
									val_loss += val_loss_
									val_acc += val_acc_
									val_steps += 1
								except:
									break
							sess.run(val_dataset.initializer)
							sess.run(train_dataset.initializer)
							val_loss /= val_steps
							val_acc /= val_steps
							print('\n')
							if val_acc >= best_val_acc:
								checkpoint_path = os.path.join(self.log_path, 'model.ckpt')
								saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
								print('accuracy improved from %.3f to %.3f' % (best_val_acc, val_acc))
								sys.stdout.flush()  # 刷新缓冲区
								best_val_acc = val_acc
							break
			sess.close()


if __name__ == '__main__':
    tarin_model = TrainModel()
    tarin_model.train()

