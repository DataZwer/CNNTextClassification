import os
import model
import tensorflow as tf
import data_utils
import configuration

#加载数据，标签，词对应的索引
data, labels, w2idx = data_utils.get_data(configuration.config['paths'])

#词的总数
configuration.config['n_words'] = len(w2idx)+1

#建立TensorFlow会话
with tf.Session() as sess:
    #模型构建
    net = model.CNN(configuration.config, sess)
    #模型训练
    net.train(data, labels)
    #模型验证
    #....
    #模型测试
    net.test(data, labels)