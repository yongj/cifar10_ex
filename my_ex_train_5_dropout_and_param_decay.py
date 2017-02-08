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
BAT_SIZE_TRANING = 128
NUM_CLASSES = 10

BAT_SIZE_TEST = 10000

# KEEP_PROB = 0.9

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EPOCHS_PER_DECAY = 150.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-4       # Initial learning rate.

def train_nn(restart=True):
    
    #global_step = tf.contrib.framework.get_or_create_global_step()
    global_step = tf.Variable(0, trainable=False)
    
    
    x = tf.placeholder(tf.float32,[None, HEIGHT, WIDTH, DEPTH])
    labels = tf.placeholder(tf.float32,[None,NUM_CLASSES])
    bat_size = tf.placeholder(tf.int32)
    keep_prob = tf.placeholder(tf.float32)
    
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
    pool2_reshape = tf.reshape(pool2,[bat_size,-1])
    dim = 4096#102400#pool2_reshape.get_shape()[1].value
    w_local3 = tf.Variable(tf.truncated_normal(shape=[dim,384], stddev=0.1),name='w_local3')
    b_local3 = tf.Variable(tf.constant(0.1,shape=[384]),name='b_local3')
    local3 = tf.nn.relu(tf.matmul(pool2_reshape,w_local3)+b_local3)
    
    # apply dropout after local3
    dropped_local3 = tf.nn.dropout(local3, keep_prob)
    
    # local 4
    w_local4 = tf.Variable(tf.truncated_normal([384,192], stddev=0.1),name='w_local4')
    b_local4 = tf.Variable(tf.constant(0.1,shape=[192]),name='b_local4')
    local4=tf.nn.relu(tf.matmul(dropped_local3,w_local4)+b_local4)
    
    # apply dropout after local4
    dropped_local4 = tf.nn.dropout(local4, keep_prob)
    
    # softmax
    w_softmax = tf.Variable(tf.truncated_normal([192,NUM_CLASSES], stddev=0.1),name='w_softmax')
    b_softmax = tf.Variable(tf.constant(0.1,shape=[NUM_CLASSES]),name='b_softmax')
    soft_max_linear = tf.add(tf.matmul(dropped_local4,w_softmax),b_softmax)
    
    # loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(soft_max_linear,labels))
    
    # train
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BAT_SIZE_TRANING
    # decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    decay_steps = 30000
    
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                      global_step,
                                      decay_steps,
                                      LEARNING_RATE_DECAY_FACTOR,
                                      staircase=True)
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(soft_max_linear,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    test_images,test_labels = my_ex_input.get_image_label_test()

    # add summaries
    tf.summary.scalar('weight_conv1',tf.reduce_mean(w_conv1))
    tf.summary.scalar('bias_conv1',tf.reduce_mean(b_conv1))
    tf.summary.histogram('w_conv1_hist',w_conv1)
    tf.summary.histogram('b_conv1_hist',b_conv1)   
    
    tf.summary.scalar('weight_conv2',tf.reduce_mean(w_conv2))
    tf.summary.scalar('bias_conv2',tf.reduce_mean(b_conv2))
    tf.summary.histogram('w_conv2_hist',w_conv2)
    tf.summary.histogram('b_conv2_hist',b_conv2)   
    
    tf.summary.scalar('weight_local3',tf.reduce_mean(w_local3))
    tf.summary.scalar('bias_local3',tf.reduce_mean(b_local3))
    tf.summary.histogram('w_local3_hist',w_local3)
    tf.summary.histogram('b_local3_hist',b_local3)   
    
    tf.summary.scalar('weight_local4',tf.reduce_mean(w_local4))
    tf.summary.scalar('bias_local4',tf.reduce_mean(b_local4))
    tf.summary.histogram('w_local4_hist',w_local4)
    tf.summary.histogram('b_local4_hist',b_local4)   
    
    tf.summary.scalar('weight_softmax',tf.reduce_mean(w_softmax))
    tf.summary.scalar('bias_softmax',tf.reduce_mean(b_softmax))
    tf.summary.histogram('w_softmax',w_softmax)
    tf.summary.histogram('b_softmax',b_softmax)
    
    tf.summary.scalar('soft_max_linear',tf.reduce_mean(soft_max_linear))
    tf.summary.histogram('soft_max_linear_hist',soft_max_linear)
    
    tf.summary.scalar('loss',loss)
    tf.summary.scalar('accuracy',accuracy)
    tf.summary.scalar('learning_rate',lr)      
     
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/cifar10_my_ex/train_try')
    test_writer = tf.summary.FileWriter('/tmp/cifar10_my_ex/test_try')
    
    saver = tf.train.Saver()    # add the option to save training parameters  
    
    with tf.Session() as sess:
        if restart:
            sess.run(tf.global_variables_initializer())
            print('\n======== New checkpoint created in tmp/cifar10_my_excheckpoint/ ===========' )
        else:
            saver.restore(sess,'/tmp/cifar10_my_ex/checkpoint/cifar10_param.ckpt')
            print('\n======== Model restored from tmp/cifar10_my_ex/checkpoint/ ===========' )
        for i in range(300000):
            train_images, train_labels = my_ex_input.get_image_label_batch(bat_size = BAT_SIZE_TRANING)
            _ = sess.run([train_step],feed_dict={x:train_images,labels:train_labels,bat_size:BAT_SIZE_TRANING,keep_prob:0.9})
            if (i%100==0):           # evaluate on test data every 100 steps 
                test_loss, test_acc, summary = sess.run([loss,accuracy,merged],feed_dict={x:test_images,labels:test_labels,bat_size:BAT_SIZE_TEST,keep_prob:1})
                test_writer.add_summary(summary,i)
                print('\n ======== step %d, test losss: %f test accuracy: %f ===========' % (i, test_loss, test_acc))
                save_path = saver.save(sess,'/tmp/cifar10_my_ex/checkpoint/cifar10_param.ckpt')        # save the parameters to a local folder
                print('\n ======== Model saved in tmp/cifar10_my_ex/checkpoint/ ===========')
            if (i%10==0):           # evaluate on batch data every 20 steps
                output_loss = sess.run([loss],feed_dict={x:train_images,labels:train_labels,bat_size:BAT_SIZE_TRANING,keep_prob:0.9})
                output_acc,summary = sess.run([accuracy,merged],feed_dict={x:train_images,labels:train_labels,bat_size:BAT_SIZE_TRANING,keep_prob:1})
                #summary = sess.run([merged],feed_dict={x:train_images,labels:train_labels})
                train_writer.add_summary(summary,i)
                print('\n step %d, loss: %f, batch accuracy:%f' % (i,1.0,output_acc))
        train_writer.close()
        test_writer.close()
        


train_nn(restart=True)
            

            