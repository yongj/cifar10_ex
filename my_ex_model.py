# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 08:10:01 2017

@author: jiang_y
"""
import tensorflow as tf

# may need to get image size by shape later
HEIGHT_DISTORTED = 24
WIDTH_DISTORTED = 24

NUM_CLASSES = 10
BAT_SIZE_TRANING = 128

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EPOCHS_PER_DECAY = 150.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-4       # Initial learning rate.

def inference(training_images,bat_size,keep_prob):
    '''
    Build Deep CNN Model. bat_size and keep_prob needs to be passed in because it is differnt for taining/test sets
    Input: training images, bat_size, keep_prob
    Output: logits
    '''
    with tf.name_scope('conv1') as scope:       # conv1
        w_conv1 = tf.Variable(tf.truncated_normal([5,5,3,64], stddev=0.1),name='w_conv1')
        b_conv1 = tf.Variable(tf.constant(0.1,shape=[64]),name='b_conv1')    
        conv = tf.nn.conv2d(training_images,w_conv1,[1,1,1,1],padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, b_conv1),name='conv1')
        
        tf.summary.scalar('weight_conv1',tf.reduce_mean(w_conv1))
        tf.summary.scalar('bias_conv1',tf.reduce_mean(b_conv1))
        tf.summary.histogram('w_conv1_hist',w_conv1)
        tf.summary.histogram('b_conv1_hist',b_conv1) 
        
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME',name='pool1')
    
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75,name='norm1')
    
    with tf.name_scope('conv2') as scope:       # conv2
        w_conv2 = tf.Variable(tf.truncated_normal([5,5,64,64], stddev=0.1),name='w_conv2')
        b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]),name='b_conv2')
        conv = tf.nn.conv2d(norm1,w_conv2,[1,1,1,1],padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, b_conv2),name='conv2')
        
        tf.summary.scalar('weight_conv2',tf.reduce_mean(w_conv2))
        tf.summary.scalar('bias_conv2',tf.reduce_mean(b_conv2))
        tf.summary.histogram('w_conv2_hist',w_conv2)
        tf.summary.histogram('b_conv2_hist',b_conv2)   
    
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75,name='norm2')
    
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME',name='pool2')
    
    with tf.name_scope('local3') as scope:   # local3
        pool2_reshape = tf.reshape(pool2,[bat_size,-1])
        dim = int(HEIGHT_DISTORTED*WIDTH_DISTORTED*64/4/4)  #4096#102400#pool2_reshape.get_shape()[1].value
        w_local3 = tf.Variable(tf.truncated_normal(shape=[dim,384], stddev=0.1),name='w_local3')
        b_local3 = tf.Variable(tf.constant(0.1,shape=[384]),name='b_local3')
        local3 = tf.nn.relu(tf.matmul(pool2_reshape,w_local3)+b_local3,name='local3')
                
        tf.summary.scalar('weight_local3',tf.reduce_mean(w_local3))
        tf.summary.scalar('bias_local3',tf.reduce_mean(b_local3))
        tf.summary.histogram('w_local3_hist',w_local3)
        tf.summary.histogram('b_local3_hist',b_local3)   
    
        # apply dropout after local3
        dropped_local3 = tf.nn.dropout(local3, keep_prob,name='dropout_local3')
    
    with tf.name_scope('local4') as scope:   # local 4
        w_local4 = tf.Variable(tf.truncated_normal([384,192], stddev=0.1),name='w_local4')
        b_local4 = tf.Variable(tf.constant(0.1,shape=[192]),name='b_local4')
        local4=tf.nn.relu(tf.matmul(dropped_local3,w_local4)+b_local4,name='local4')
        
        # apply dropout after local4
        dropped_local4 = tf.nn.dropout(local4, keep_prob,name='dropout_local4')
        
        tf.summary.scalar('weight_local4',tf.reduce_mean(w_local4))
        tf.summary.scalar('bias_local4',tf.reduce_mean(b_local4))
        tf.summary.histogram('w_local4_hist',w_local4)
        tf.summary.histogram('b_local4_hist',b_local4)   
    
    with tf.name_scope('softmax') as scope:   # softmax
        w_softmax = tf.Variable(tf.truncated_normal([192,NUM_CLASSES], stddev=0.1),name='w_softmax')
        b_softmax = tf.Variable(tf.constant(0.1,shape=[NUM_CLASSES]),name='b_softmax')
        soft_max_linear = tf.add(tf.matmul(dropped_local4,w_softmax),b_softmax,name='logits')
        
        tf.summary.scalar('weight_softmax',tf.reduce_mean(w_softmax))
        tf.summary.scalar('bias_softmax',tf.reduce_mean(b_softmax))
        tf.summary.histogram('w_softmax',w_softmax)
        tf.summary.histogram('b_softmax',b_softmax)
        
        tf.summary.scalar('soft_max_linear',tf.reduce_mean(soft_max_linear))
        tf.summary.histogram('soft_max_linear_hist',soft_max_linear)
        
    return soft_max_linear
    
def loss(logits,labels):
    '''
    Contruct the loss as the softmax cross entropy of logits and labels
    Input labels need to be one-hot vector
    '''
    labels = tf.to_int64(labels)
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,labels,name='xentropy')
        loss = tf.reduce_mean(cross_entropy,name='xentropy_mean')
        tf.summary.scalar('loss',loss)
    return loss
    
def training(loss,step_i):
    '''
    Construct training op
    Need step_i input to implement learning rate decay
    '''
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BAT_SIZE_TRANING
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)     # decay_steps = 58593
    decay_steps = 1000000       # hack the learning rate to be fixed
    
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                      step_i,
                                      decay_steps,
                                      LEARNING_RATE_DECAY_FACTOR,
                                      staircase=True)
    tf.summary.scalar('learning_rate',lr) 
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    return train_op
    
def evaluation(logits,labels):
    '''
    evaluate the accuracy
    '''
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy',accuracy)
    return accuracy