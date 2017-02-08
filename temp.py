# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 08:00:49 2017

@author: jiang_y
"""
import my_ex_input
import os
from six.moves import cPickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

img,label = my_ex_input.get_image_label_batch(bat_size = 128)
i = np.random.randint(0,128)

img_tf = tf.convert_to_tensor(img)
#img = tf.slice(img_tf,[i,0,0,0],[1,32,32,3])



i = np.random.randint(0,128)
a=img[i,:,:,:]
a = a.astype(np.uint8)
plt.imshow(a)

d=a/255
plt.imshow(d)

c=a
r,g,b = (0,1,2)
c[:,:,0] = a[:,:,r]
c[:,:,1] = a[:,:,g]
c[:,:,2] = a[:,:,b]
plt.imshow(c)


b = a* 0.5
b = b.astype(np.uint8)
plt.imshow(b)




