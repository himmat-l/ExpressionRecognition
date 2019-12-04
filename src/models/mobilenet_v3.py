"""Implementation of Mobilenet V3.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import os
import cv2
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
tf.reset_default_graph()


class MobileNet_v3(object):
	def __init__(self, batch_size=32, class_num=8, multiplier=1.0):
		self.batch_size = batch_size
		self.class_num = class_num
		self.multiplier = multiplier
		self.batch_norm_params = {
			'momentum': 0.997,
			'epsilon': 0.0001,
			'is_training': False
		}

	def make_divisible(self, v, divisor=8, min_value=None):
		if min_value is None:
			min_value = divisor
		new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
		# Make sure that round down does not go down by more than 10%.
		if new_v < 0.9 * v:
			new_v += divisor
		return new_v

	def relu(self, x, name='relu'):
		return tf.nn.relu(x, name)

	def relu6(self, x, name='relu6'):
		with tf.variable_scope(name):
			return tf.nn.relu6(x, name)

	def hard_swish(self, x, name='hard_swish'):
		with tf.variable_scope(name):
			h_swish = x * tf.nn.relu6(x + 3) / 6
		return h_swish

	def hard_sigmoid(self, x, name='hard_sigmoid'):
		with tf.variable_scope(name):
			h_sigmoid = tf.nn.relu6(x + 3) / 6
		return h_sigmoid

	def bn_layer(self, inputs, name, reuse=None):
		return tf.layers.batch_normalization(inputs,
		                                  momentum=self.batch_norm_params['momentum'],
		                                  epsilon=self.batch_norm_params['epsilon'],
		                                  training=self.batch_norm_params['is_training'],
		                                  name=name, reuse=reuse)

	def conv_1x1_bn(self, inputs, filters_num, name, use_bias=True, reuse=None):
		x = tf.layers.conv2d(inputs,
		                     filters=filters_num, kernel_size=1, strides=1, padding='SAME',
		                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3),
		                     use_bias=use_bias, reuse=reuse)
		x = self.bn_layer(x, name=name+'/bn',reuse=reuse)
		return x

	def conv_dwise(self, inputs,
	               kernel_size=3,
	               depth_multiplier=1,
	               strides=1,
	               padding='SAME',
	               name='dwise_conv',
	               use_bias=False,
	               reuse=None):
		input_channel = inputs.get_shape().as_list()[-1]
		filters = int(input_channel*depth_multiplier)
		outputs = tf.layers.separable_conv2d(inputs,
		                                     filters,
		                                     kernel_size,
		                                     strides=strides,
		                                     padding=padding,
		                                     data_format='channels_last', depth_multiplier=depth_multiplier,
		                                     activation=None, use_bias=use_bias, name=name, reuse=reuse)
		return outputs

	def conv_norm_act(self, inputs, filters_num, name,
	                  kernel_size=3,
	                  strides=1,
	                  padding='SAME',
	                  activation='HS',
	                  use_bias=True,
	                  reuse=None,
	                  l2_reg=1e-5):

		x = tf.layers.conv2d(inputs,
		                     filters=filters_num, kernel_size=kernel_size, strides=strides, padding=padding,
		                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
		                     use_bias=use_bias)
		x = self.bn_layer(x, name=name+'/bn')
		if activation == 'HS':
			x = self.hard_swish(x)
		elif activation == 'RE':
			x = self.relu6(x)
		else:
			raise NotImplementedError
		return x

	def se_bneck(self, inputs, out_dim, ratio, name="SEBottleneck", reuse=None):
		squeeze = tf.layers.average_pooling2d(inputs, pool_size=inputs.get_shape()[1:-1], strides=1)
		excitation = tf.layers.dense(squeeze, units=out_dim/ratio, name=name+'_excitation1', reuse=reuse)
		excitation = self.relu6(excitation)
		excitation = tf.layers.dense(excitation, units=out_dim, name=name+'_excitation2', reuse=reuse)
		excitation = self.hard_sigmoid(excitation)
		excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
		scale = inputs * excitation
		return scale

	def mobilenet_v3_block(self, inputs, out_channels, exp_channels, kernel_size, stride, name, ratio=16,
	          use_bias=True, shortcut=True, activation='RE', use_se=False,  reuse=None):
		with tf.variable_scope(name, reuse=None):
			# pointwise
			net = self.conv_1x1_bn(inputs, exp_channels, name='pointwise', use_bias=use_bias)
			if activation == 'HS':
				net = self.hard_swish(net)
			elif activation == "RE":
				net = self.relu6(net)
			else:
				raise NotImplementedError

			# depthwise
			net = self.conv_dwise(net, kernel_size=kernel_size, strides=stride, name='depthwise', use_bias=use_bias, reuse=reuse)
			net = self.bn_layer(net, name='dw_bn', reuse=reuse)
			if activation == 'HS':
				net = self.hard_swish(net)
			elif activation == "RE":
				net = self.relu6(net)
			else:
				raise NotImplementedError

			# squeeze and excitation
			if use_se:
				channel = net.get_shape().as_list()[-1]
				net = self.se_bneck(net, out_dim=channel, ratio=ratio, name='se_bneck')

			# pointwise and linear
			net = self.conv_1x1_bn(net, out_channels, name='pw_linear', use_bias=use_bias)

			# element wise add, only for stride==1
			if shortcut and stride == 1:
				net += inputs
				net = tf.identity(net, name='block_output')
			return net

	def build_model_small(self, inputs, reduction_ratio, reuse=None):
		end_points = {}
		layers = [
			# in_chn  out_chn k_s stride act se exp_size
			[16, 16, 3, 2, "RE", True, 16],
			[16, 24, 3, 2, "RE", False, 72],
			[24, 24, 3, 1, "RE", False, 88],
			[24, 40, 5, 2, "RE", True, 96],
			[40, 40, 5, 1, "RE", True, 240],
			[40, 40, 5, 1, "RE", True, 240],
			[40, 48, 5, 1, "HS", True, 120],
			[48, 48, 5, 1, "HS", True, 144],
			[48, 96, 5, 2, "HS", True, 288],
			[96, 96, 5, 1, "HS", True, 576],
			[96, 96, 5, 1, "HS", True, 576],
		]
		input_size = inputs.get_shape().as_list()[1:-1]  # 输入特征尺寸
		assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))

		with tf.variable_scope('init', reuse=reuse):
			init_conv_out = self.make_divisible(16*self.multiplier)
			x = self.conv_norm_act(inputs, filters_num=init_conv_out, kernel_size=3, name='init_layer',
			                       strides=2, activation='HS', use_bias=False)

		with tf.variable_scope('Mobilenetv3_small', reuse=reuse):
			for idx, (in_channels, out_channels, kernel_size, stride, activation, se, exp_size) in enumerate(layers):
				in_channels = self.make_divisible(in_channels*self.multiplier)
				out_channels = self.make_divisible(out_channels*self.multiplier)
				exp_channels = self.make_divisible(exp_size*self.multiplier)

				x = self.mobilenet_v3_block(x, out_channels, exp_channels, kernel_size, stride,
				                             name="bneck{}".format(idx), use_bias=True, shortcut=(in_channels == out_channels),
				                             activation=activation, ratio=reduction_ratio)
				end_points['bneck{}'.format(idx)] = x

			conv1_out = self.make_divisible(576*self.multiplier)
			x = self.conv_norm_act(x, conv1_out, name='conv1_out', kernel_size=1, strides=1, activation='HS', use_bias=True)
			x = self.se_bneck(x, conv1_out, reduction_ratio, name='conv1_out_se')
			end_points['conv1_out_1x1'] = x

			x = tf.layers.average_pooling2d(x, pool_size=x.get_shape()[1:-1], strides=1, name='global_avg')
			end_points['global_avg'] = x

		with tf.variable_scope('logits_out', reuse=reuse):
			conv2_out = self.make_divisible(1280*self.multiplier)
			x = tf.layers.conv2d(x, filters=conv2_out, kernel_size=1, strides=1, padding='SAME', name='conv2', use_bias=True)
			x = self.hard_swish(x)
			end_points['conv2_out_1x1'] = x

			x = tf.layers.conv2d(x, filters=self.class_num, kernel_size=1, strides=1, padding='SAME', name='conv3', use_bias=True)
			logits = tf.layers.flatten(x)
			logits = tf.identity(logits, name='output')
			end_points['Logits_out'] = logits
		return logits, end_points





if __name__ == '__main__':
	with tf.Session() as sess:
		input_test = tf.zeros([1, 224, 224, 3])
		input_img = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
		image = cv2.imread(r'C:\Users\user\Pictures\Screenshots\1.jpg')
		image = cv2.resize(image, (224, 224))
		image = np.array(image, dtype=np.float).reshape((1,224,224,3))
		print(image.shape[1])
		#inputs = tf.constant([[[[1, 3, 3], [7, 5, 6]], [[7, 8, 9], [10, 11, 15]]]], dtype=tf.float32)
		# inputs_shape=tf.Tensor.get_shape(inputs).as_list()
		model = MobileNet_v3()
		model, end_points = model.build_model_small(input_test, reduction_ratio=4, reuse=None)
		sess.run(tf.global_variables_initializer())
		print(sess.run(model))
