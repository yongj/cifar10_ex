# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 08:09:53 2016

@author: jiang_y
"""

import tensorflow as tf
import my_ex_input

HEIGHT = 32
WIDTH = 32
DEPTH = 3
BAT_SIZE = 128
NUM_CLASSES = 10

TEST_SAMPLES = 10000

def train_nn():
    x = tf.placeholder(tf.float32,[None, HEIGHT, WIDTH, DEPTH])
    labels = tf.placeholder(tf.float32,[None,NUM_CLASSES])
    
    # conv1
    w_conv1 = tf.Variable(tf.truncated_normal([5,5,3,64], stddev=0.1),name='w_conv1')
    b_conv1 = tf.Variable(tf.constant(0.1,shape=[64]),name='b_conv1')    
    conv = tf.nn.conv2d(x,w_conv1,[1,1,1,1],padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv, b_conv1))
    
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    
    # conv2
    w_conv2 = tf.Variable(tf.truncated_normal([5,5,64,64], stddev=0.1),name='w_conv2')
    b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]),name='b_conv2')
    conv = tf.nn.conv2d(norm1,w_conv2,[1,1,1,1],padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv, b_conv2))
    
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    
    # local3
    pool2_reshape = tf.reshape(pool2,[BAT_SIZE,-1])
    dim = 4096#102400#pool2_reshape.get_shape()[1].value
    w_local3 = tf.Variable(tf.truncated_normal(shape=[dim,384], stddev=0.1),name='w_local3')
    b_local3 = tf.Variable(tf.constant(0.1,shape=[384]),name='b_local3')
    local3 = tf.nn.relu(tf.matmul(pool2_reshape,w_local3)+b_local3)
    
    # local 4
    w_local4 = tf.Variable(tf.truncated_normal([384,192], stddev=0.1),name='w_local4')
    b_local4 = tf.Variable(tf.constant(0.1,shape=[192]),name='b_local4')
    local4=tf.nn.relu(tf.matmul(local3,w_local4)+b_local4)
    
    # softmax
    w_softmax = tf.Variable(tf.truncated_normal([192,NUM_CLASSES], stddev=0.1),name='w_softmax')
    b_softmax = tf.Variable(tf.constant(0.1,shape=[NUM_CLASSES]),name='b_softmax')
    soft_max_linear = tf.add(tf.matmul(local4,w_softmax),b_softmax)
    
    # loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(soft_max_linear,labels))
    
    # train
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(soft_max_linear,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # get test data for evaluation. put it outside the loop so that data only pulled once
    # local3
    pool2_reshape_test = tf.reshape(pool2,[TEST_SAMPLES,-1])
    local3_test = tf.nn.relu(tf.matmul(pool2_reshape_test,w_local3)+b_local3)
    
    # local 4
    local4_test=tf.nn.relu(tf.matmul(local3_test,w_local4)+b_local4)
    
    # softmax
    soft_max_linear_test = tf.add(tf.matmul(local4_test,w_softmax),b_softmax)
    
    correct_prediction_test = tf.equal(tf.argmax(soft_max_linear_test,1), tf.argmax(labels,1))
    accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))
    test_images,test_labels = my_ex_input.get_image_label_test()
    
    saver = tf.train.Saver()    # add the option to save training parameters
    
    with tf.Session() as sess:
        # sess = tf.InteractiveSession()
        # tf.global_variables_initializer().run()
        
        saver.restore(sess,'/tmp/cifar10_my_ex/checkpoint/cifar10_param.ckpt')
        print('\n======== Model restored from tmp/cifar10_my_ex ===========' )
        for i in range(100000):
            train_images, train_labels = my_ex_input.get_image_label_batch(bat_size = BAT_SIZE)
            _ = sess.run([train_step],feed_dict={x:train_images,labels:train_labels})
            if (i%1000==0):           # evaluate on test data every 100 steps 
                test_acc = accuracy_test.eval(feed_dict={x:test_images,labels:test_labels})
                print('\n ======== step %d, test accuracy: %f ===========' % (i, test_acc))
                save_path = saver.save(sess,'/tmp/cifar10_my_ex/cifar10_param.ckpt')        # save the parameters to a local folder
                print('\n ======== Model saved in tmp/cifar10_my_ex ===========')
            if (i%100==0):           # evaluate on batch data every 20 steps
                output_loss,output_acc = sess.run([loss,accuracy],feed_dict={x:train_images,labels:train_labels})
                print('\n step %d, loss: %f, batch accuracy:%f' % (i,output_loss,output_acc))
        


train_nn()
            

            