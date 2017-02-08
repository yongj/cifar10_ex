# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 08:09:53 2016

@author: jiang_y
"""

import tensorflow as tf
import my_ex_input as input
import my_ex_model as model
import my_ex_send_email as send_email
import time

HEIGHT = 32
WIDTH = 32
DEPTH = 3

HEIGHT_DISTORTED = 24
WIDTH_DISTORTED = 24

BAT_SIZE_TRANING = 128
NUM_CLASSES = 10

BAT_SIZE_TEST = 10000

# KEEP_PROB = 0.9

def train_nn(restart=True):
    test_images,test_labels = input.get_image_label_test_distorted()
    with tf.Graph().as_default():
        # define the placeholders for feed_dict
        step_i = tf.placeholder(tf.int32)
        images = tf.placeholder(tf.float32,[None, HEIGHT_DISTORTED, WIDTH_DISTORTED, DEPTH])
        labels = tf.placeholder(tf.float32,[None,NUM_CLASSES])
        bat_size = tf.placeholder(tf.int32)
        keep_prob = tf.placeholder(tf.float32)
        
        logits = model.inference(images,bat_size,keep_prob)
        loss = model.loss(logits,labels)        # loss
        train_op = model.training(loss,step_i)      # train        
        accuracy = model.evaluation(logits,labels)
    
        merged = tf.summary.merge_all()      # merge summmeries
        
        saver = tf.train.Saver()    # add the option to save training parameters  
        
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('/tmp/cifar10_my_ex/train_try',sess.graph)
            test_writer = tf.summary.FileWriter('/tmp/cifar10_my_ex/test_try')
            if restart:
                sess.run(tf.global_variables_initializer())
                print('\n======== New checkpoint created in tmp/cifar10_my_excheckpoint/ ===========' )
            else:
                saver.restore(sess,'/tmp/cifar10_my_ex/checkpoint/cifar10_param.ckpt')
                print('\n======== Model restored from tmp/cifar10_my_ex/checkpoint/ ===========' )
                
            t0 = time.time()
            
            for i in range(300000):
                if i not in [999,4999,9999,14999]:
                    t1 = time.time()
                    train_images, train_labels = input.get_image_label_batch_distorted(bat_size = BAT_SIZE_TRANING)
                    t2 = time.time()
                    _ = sess.run([train_op],feed_dict={images:train_images,labels:train_labels,bat_size:BAT_SIZE_TRANING,keep_prob:1.0,step_i:i})
                    t = time.time()
                    if (i%100==0):           # evaluate on test data every 100 steps 
                        test_loss, test_acc, summary = sess.run([loss,accuracy,merged],feed_dict={images:test_images,labels:test_labels,bat_size:BAT_SIZE_TEST,keep_prob:1,step_i:i})
                        test_writer.add_summary(summary,i)
                        print('\n ======== step %d, test losss: %f test accuracy: %f ===========' % (i, test_loss, test_acc))
                        save_path = saver.save(sess,'/tmp/cifar10_my_ex/checkpoint/cifar10_param.ckpt')        # save the parameters to a local folder
                        print('\n ======== Model saved in tmp/cifar10_my_ex/checkpoint/ ===========')
                    if (i%10==0):           # evaluate on batch data every 20 steps                    
                        output_loss = sess.run([loss],feed_dict={images:train_images,labels:train_labels,bat_size:BAT_SIZE_TRANING,keep_prob:1.0,step_i:i})
                        output_acc,summary = sess.run([accuracy,merged],feed_dict={images:train_images,labels:train_labels,bat_size:BAT_SIZE_TRANING,keep_prob:1,step_i:i})
                        #summary = sess.run([merged],feed_dict={x:train_images,labels:train_labels})
                        train_writer.add_summary(summary,i)
                        print('\n step %d, loss: %.2f, bat accu:%.3f, tot time: %.1f, avg time: %.3f, last time:%.3f, last load data time %.3f'
                              % (i,output_loss[0],output_acc,t-t0,(t-t0)/(i+1),t-t1,t2-t1))
                    if (i%5000==0):      # send me an email every 100 steps
                        message1 = format('Model 8 update => test accuracy: %.2f, training step: %d, batch accuracy: %.2f\n' % (test_acc,i,output_acc))
                        measage2 = format('\n step %d, loss: %.2f, bat accu:%.3f, tot time: %.1f, avg time: %.3f, last time:%.3f, last load data time %.3f'
                              % (i,output_loss[0],output_acc,t-t0,(t-t0)/(i+1),t-t1,t2-t1))
                        send_email.send_email('Accuracy Update for Model 8',message1+measage2)
                else:       # log runtime statistics
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _,summary = sess.run([train_op,merged],feed_dict={images:train_images,labels:train_labels,bat_size:BAT_SIZE_TRANING,keep_prob:1.0,step_i:i},
                                         options=run_options,
                                         run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)
            train_writer.close()
            test_writer.close()
            


train_nn(restart=True)
            

            