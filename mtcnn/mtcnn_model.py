#coding:utf-8
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
num_keep_radio = 0.7
#define prelu
def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg

def dense_to_one_hot(labels_dense,num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    #num_sample*num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
#cls_prob:batch*2
#label:batch

def cls_ohem(cls_prob, label):
    zeros = tf.zeros_like(label)
    #label=-1 --> label=0net_factory

    #pos -> 1, neg -> 0, others -> 0
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int = tf.cast(label_filter_invalid,tf.int32)
    # get the number of rows of class_prob
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    #row = [0,2,4.....]
    row = tf.range(num_row)*2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))  # tf.gather：用一个一维的索引数组，将张量中对应索引的向量提取出来
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    # set pos and neg to be 1, rest to be 0
    valid_inds = tf.where(label < zeros,zeros,ones)
    # get the number of POS and NEG examples
    num_valid = tf.reduce_sum(valid_inds)

    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #FILTER OUT PART AND LANDMARK DATA
    loss = loss * valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num) # tf.nn.top_k：这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
    return tf.reduce_mean(loss)


def bbox_ohem_smooth_L1_loss(bbox_pred,bbox_target,label):
    sigma = tf.constant(1.0)
    threshold = 1.0/(sigma**2)
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    abs_error = tf.abs(bbox_pred-bbox_target)
    loss_smaller = 0.5*((abs_error*sigma)**2)
    loss_larger = abs_error-0.5/(sigma**2)
    smooth_loss = tf.reduce_sum(tf.where(abs_error<threshold,loss_smaller,loss_larger),axis=1)
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    smooth_loss = smooth_loss*valid_inds
    _, k_index = tf.nn.top_k(smooth_loss, k=keep_num)
    smooth_loss_picked = tf.gather(smooth_loss, k_index)
    return tf.reduce_mean(smooth_loss_picked)
def bbox_ohem_orginal(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    #pay attention :there is a bug!!!!
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    #(batch,)
    square_error = tf.reduce_sum(tf.square(bbox_pred-bbox_target),axis=1)
    #keep_num scalar
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

#label=1 or label=-1 then do regression
def bbox_ohem(bbox_pred,bbox_target,label):
    '''

    :param bbox_pred:
    :param bbox_target:
    :param label: class label
    :return: mean euclidean loss for all the pos and part examples
    '''
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)
    # keep pos and part examples
    valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)
    #(batch,)
    #calculate square sum
    square_error = tf.square(bbox_pred-bbox_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    #keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    # count the number of pos and part examples
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    # keep top k examples, k equals to the number of positive examples
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)

    return tf.reduce_mean(square_error)

def landmark_ohem(landmark_pred,landmark_target,label):
    '''

    :param landmark_pred:
    :param landmark_target:
    :param label:
    :return: mean euclidean loss
    '''
    #keep label =-2  then do landmark detection
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)
    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
    
def cal_accuracy(cls_prob,label):
    '''

    :param cls_prob:
    :param label:
    :return:calculate classification accuracy for pos and neg examples only
    '''
    # get the index of maximum value along axis one from cls_prob
    # 0 for negative 1 for positive
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(label,tf.int64)
    # return the index of pos and neg examples
    cond = tf.where(tf.greater_equal(label_int,0))
    picked = tf.squeeze(cond)
    # gather the label of pos and neg examples
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    #calculate the mean value of a vector contains 1 and 0, 1 for correct classification, 0 for incorrect
    # ACC = (TP+FP)/total population
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op


def _activation_summary(x):
    '''
    creates a summary provides histogram of activations
    creates a summary that measures the sparsity of activations

    :param x: Tensor
    :return:
    '''

    tensor_name = x.op.name
    print('load summary for : ',tensor_name)
    tf.summary.histogram(tensor_name + '/activations',x)
    #tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))





#construct Pnet
#label:batch
def P_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    #define common param
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005), 
                        padding='valid'):
        print(inputs.get_shape())


        net = slim.conv2d(inputs, 10, 3, stride=1,scope='conv1')
        _activation_summary(net)
        print(net.get_shape())
        # 通过10个3*3*3的卷积核，2*2的Max Pooling（stride=2）操作，生成10个5*5的特征图
        net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope='pool1', padding='SAME')
        _activation_summary(net)
        print(net.get_shape())
        # 通过16个3*3*10的卷积核，生成16个3*3的特征图
        net = slim.conv2d(net,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')
        _activation_summary(net)
        print(net.get_shape())
        # 通过32个3*3*16的卷积核，生成32个1*1的特征图
        net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')
        _activation_summary(net)
        print(net.get_shape())
        # batch*H*W*2，针对32个1*1的特征图，可以通过2个1*1*32的卷积核，生成2个1*1的特征图用于分类
        conv4_1 = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
        _activation_summary(conv4_1)
        # conv4_1 = slim.conv2d(net,num_outputs=1,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.sigmoid)
        
        print (conv4_1.get_shape())
        # batch*H*W*4，4个1*1*32的卷积核，生成4个1*1的特征图用于回归框判断
        bbox_pred = slim.conv2d(net,num_outputs=4,kernel_size=[1,1],stride=1,scope='conv4_2',activation_fn=None)
        _activation_summary(bbox_pred)
        print (bbox_pred.get_shape())
        #batch*H*W*10，10个1*1*32的卷积核，生成10个1*1的特征图用于人脸轮廓点的判断
        landmark_pred = slim.conv2d(net,num_outputs=10,kernel_size=[1,1],stride=1,scope='conv4_3',activation_fn=None)
        _activation_summary(landmark_pred)
        print (landmark_pred.get_shape())


        # add projectors for visualization



    # PNet的训练数据主要由4部分组成，包括：
    # 正label数据(IOU>0.65,面部轮廓特值为0)，负label数据(IOU<0.4，面部轮廓特值为0，回归框值为0)，中间数据（0.4<IOU<0.65，面部轮廓特值为0），面部轮廓数据（回归框值为0）。
    # 把训练数据输入到网络中后，依据网络输出，计算loss值

        #cls_prob_original = conv4_1 
        #bbox_pred_original = bbox_pred
        if training:
            #batch*2
            # calculate classification loss， 分类loss值
            cls_prob = tf.squeeze(conv4_1,[1,2],name='cls_prob')
            cls_loss = cls_ohem(cls_prob,label)
            #batch
            # cal bounding box error, squared sum error，回归框loss
            bbox_pred = tf.squeeze(bbox_pred,[1,2],name='bbox_pred')  # tf.squeeze：该函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果
                                                                      # axis可以用来指定要删掉的为1的维度
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            #batch*10，面部轮廓loss
            landmark_pred = tf.squeeze(landmark_pred,[1,2],name="landmark_pred")
            landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)

            accuracy = cal_accuracy(cls_prob,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        #test
        else:
            #when test,batch_size = 1
            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            bbox_pred_test = tf.squeeze(bbox_pred,axis=0)
            landmark_pred_test = tf.squeeze(landmark_pred,axis=0)
            return cls_pro_test,bbox_pred_test,landmark_pred_test

# RNet主要对PNet生成的bounding box坐标进行进一步的调整。
def R_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        print (inputs.get_shape())
        # 模型输入为24*24大小的图片，通过28个3*3*3的卷积核和3*3（stride=2）的max pooling后生成28个11*11的特征图；
        net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3,3], stride=1, scope="conv1")
        print (net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print(net.get_shape())
        # 通过48个3*3*28的卷积核和3*3（stride=2）的max pooling后生成48个4*4的特征图；
        net = slim.conv2d(net,num_outputs=48,kernel_size=[3,3],stride=1,scope="conv2")
        print(net.get_shape())
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2")
        print(net.get_shape())
        # 通过64个2*2*48的卷积核后，生成64个3*3的特征图
        net = slim.conv2d(net,num_outputs=64,kernel_size=[2,2],stride=1,scope="conv3")
        print(net.get_shape())
        fc_flatten = slim.flatten(net)
        print(fc_flatten.get_shape())
        # 把3*3*64的特征图转换为128大小的全连接层
        fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1")
        print(fc1.get_shape())
        #batch*2，对回归框分类问题转换为大小为2的全连接层
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        print(cls_prob.get_shape())
        #batch*4，对bounding box的位置回归问题，转换为大小为4的全连接层
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        print(bbox_pred.get_shape())
        #batch*10，对人脸轮廓关键点转换为大小为10的全连接层
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
        print(landmark_pred.get_shape())
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)  # 分类loss
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)  # 回归框loss
            accuracy = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)  #人脸面部关键点loss
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred
# ONet是MTCNN中的最后一个网络，用于做网络的最后输出。
def O_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        print(inputs.get_shape())
        # 模型输入是一个48*48*3大小的图片，
        # 通过32个3*3*3的卷积核和3*3（stride=2）的max pooling后转换为32个23*23的特征图
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print(net.get_shape())
        # 通过64个3*3*32的卷积核和3*3（stride=2）的max pooling后转换为64个10*10的特征图
        net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], stride=1, scope="conv2")
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        print(net.get_shape())
        # 通过64个3*3*64的卷积核和3*3（stride=2）的max pooling后转换为64个4*4的特征图
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        print(net.get_shape())
        # 通过128个2*2*64的卷积核转换为128个3*3的特征图
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
        print(net.get_shape())
        fc_flatten = slim.flatten(net)
        print(fc_flatten.get_shape())
        # 通过全连接操作转换为256大小的全链接层
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1")
        print(fc1.get_shape())
        # batch*2，生成大小为2的回归框分类特征
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        print(cls_prob.get_shape())
        # batch*4，大小为4的回归框位置的回归特征
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        print(bbox_pred.get_shape())
        # batch*10，大小为10的人脸轮廓位置回归特征
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
        print(landmark_pred.get_shape())
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred