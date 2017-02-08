# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:35:54 2016

@author: jiang_y
"""
import os
from six.moves import cPickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


FOLDER_PATH = os.path.join(os.getcwd(),'cifar-10-batches-py')
HEIGHT = 32
WIDTH = 32
DEPTH = 3

HEIGHT_DISTORTED = 24
WIDTH_DISTORTED = 24
BAT_SIZE = 128

CLASS_TBL = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def pick_a_file():
    ''' randomly pick a file of training data from FOLDER_PATH '''
    file_bat_num = 5
    j = np.random.randint(1,file_bat_num+1)
    file_name = 'data_batch_'+str(j)
    file_path = os.path.join(FOLDER_PATH,file_name)
    return file_path

def unpickle(file):
    #import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo,encoding='latin1')
    fo.close()
    return dict

def get_label_tbl():    
    file_name = 'batches.meta'
    file_path = os.path.join(FOLDER_PATH,file_name)
    label_tbl = unpickle(file_path)
    return label_tbl
    
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def show_image():
    ''' Display a picture randomly.
    'filenames', 'batch_label', 'labels', 'data'
    '''
    file_path = pick_a_file()
    data = unpickle(file_path)
    
    # randomly pick a picture
    i = np.random.randint(0,data['data'].shape[0])
    img = data['data'][i,:]
    img = np.reshape(img,[DEPTH,HEIGHT,WIDTH])
    img=np.transpose(img,(1,2,0))
    
    print(plt.imshow(img))
    print(data['batch_label'])
    print(data['filenames'][i])
    print(data['labels'][i])
    
    return data['labels'][i]

def get_image_label_batch(bat_size = BAT_SIZE):
    ''' randomly get a batch file for training '''
    file_path = pick_a_file()
    data = unpickle(file_path)
    
    i = np.random.randint(0,data['data'].shape[0],bat_size)
    train_images = data['data'][i,].astype(np.float32)
    train_images = train_images.reshape([bat_size,DEPTH,HEIGHT,WIDTH])
    train_images = train_images.transpose((0,2,3,1))
    train_labels = np.asarray(data['labels'])[i]
    train_labels = dense_to_one_hot(train_labels,num_classes=10)
    return train_images, train_labels
    
def get_image_label_batch_distorted(bat_size = BAT_SIZE):
    ''' 
    randomly get a batch file for training 
    add distortion: random flip and crop
    '''
    file_path = pick_a_file()
    data = unpickle(file_path)
    
    i = np.random.randint(0,data['data'].shape[0],bat_size)
    train_images = data['data'][i,].astype(np.float32)
    train_images = train_images.reshape([bat_size,DEPTH,HEIGHT,WIDTH])
    train_images = train_images.transpose((0,3,2,1))
    if np.random.randint(0,2):  # randomly flip the image
        train_images = np.fliplr(train_images)
    j,k = np.random.randint(0,8,2)  # randomly crop the image to 24X24
    train_images = train_images[:,j:j+24,k:k+24,:]
    train_images = train_images.transpose((0,2,1,3))
    train_labels = np.asarray(data['labels'])[i]
    train_labels = dense_to_one_hot(train_labels,num_classes=10)
    return train_images, train_labels

def verify_batch(train_images,train_labels):
    bat_size = train_images.shape[0]
    i = np.random.randint(0,bat_size)
    img = train_images[i,:,:,:]
    print(plt.imshow(img))
    j = np.argmax(train_labels[i,:])
    print(CLASS_TBL[j])
    
def get_image_label_batch_distorted_tf():
    ''' 
    covert a normal image/label batch to a distorted one
    return image/label as tensors
    '''
    train_images, train_labels = get_image_label_batch(bat_size = BAT_SIZE)
    train_images_tf = tf.convert_to_tensor(train_images)
    train_labels_tf = tf.convert_to_tensor(train_labels)
    
    height = HEIGHT_DISTORTED
    width = WIDTH_DISTORTED
    
    for i in range(BAT_SIZE):
        img = tf.slice(train_images_tf,[i,0,0,0],[1,32,32,3])
        img = tf.reshape(img,[32,32,3])
        # Image processing for training the network. Note the many random
        # distortions applied to the image.
        
        # Randomly crop a [height, width] section of the image.
        img = tf.random_crop(img, [height, width, 3])
        
        # Randomly flip the image horizontally.
        # img = tf.image.random_flip_left_right(img)
        
        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        #img = tf.image.random_brightness(img, max_delta=63)
        #img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        
        # Subtract off the mean and divide by the variance of the pixels.
        #img = tf.image.per_image_standardization(img)
        
        # Set the shapes of tensors.
        img.set_shape([height, width, 3])
        #read_input.label.set_shape([1])
        
        if i==0:
            train_images_new = tf.reshape(img,[1,24,24,3])
        else:
            train_images_new = tf.concat(0,[train_images_new,tf.reshape(img,[1,24,24,3])])
    return train_images_new,train_labels_tf
        
def display_img_tensor(img):
    with tf.Session() as sess:
        b3=img.eval()
    
    print(plt.imshow(b3))        
        
    
def get_image_label_test():
    ''' load test image and labels '''
    file_path = os.path.join(FOLDER_PATH,'test_batch')
    data = unpickle(file_path)
    test_images = data['data'].astype(np.float32)
    test_images = test_images.reshape([test_images.shape[0],DEPTH,HEIGHT,WIDTH])
    test_images = test_images.transpose((0,2,3,1))
    test_labels = np.asarray(data['labels'])
    test_labels = dense_to_one_hot(test_labels,num_classes=10)
    return test_images,test_labels
    
def get_image_label_test_distorted():
    ''' 
    load test image and labels 
    crop the image to 24 * 24 in the center
    '''
    file_path = os.path.join(FOLDER_PATH,'test_batch')
    data = unpickle(file_path)
    test_images = data['data'].astype(np.float32)
    test_images = test_images.reshape([test_images.shape[0],DEPTH,HEIGHT,WIDTH])
    test_images = test_images.transpose((0,2,3,1))
    test_images = test_images[:,4:28,4:28,:]
    test_labels = np.asarray(data['labels'])
    test_labels = dense_to_one_hot(test_labels,num_classes=10)
    return test_images,test_labels

def verify_test_image(test_images,test_labels):
    i = np.random.randint(0,test_images.shape[0])
    img = test_images[i,:,:,:]
    print(plt.imshow(img))
    j = np.argmax(test_labels[i,:])
    print(CLASS_TBL[j])
    
def verify_test_image_1(test_images,test_labels):
    i = np.random.randint(0,test_images.shape[0])
    img = test_images[i,:,:,:].astype(np.uint8)
    print(plt.imshow(img))
    j = np.argmax(test_labels[i,:])
    print(CLASS_TBL[j])    

# i = show_image()
# print(CLASS_TBL[i])
# train_images, train_labels = get_image_label_batch(bat_size = 128)
# verify_batch(train_images,train_labels)
#==============================================================================
# import my_ex_input
# test_images,test_labels = my_ex_input.get_image_label_test()
# i = np.random.randint(0,test_images.shape[0])
# a = test_images[i,:,:,:]
# b = a.astype(np.uint8)
# plt.imshow(a)
# plt.imshow(b)
#==============================================================================
