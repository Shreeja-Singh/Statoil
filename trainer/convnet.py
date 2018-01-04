import tensorflow as tf
import pandas as pd
import numpy as np


def conv2d(arr, num_outputs, kernel_size):
    
    conv = tf.contrib.layers.conv2d(arr, num_outputs=num_outputs, kernel_size=kernel_size, stride = 1,
                             padding='SAME', activation_fn=tf.nn.relu, 
                             weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=42),
                             weights_regularizer=tf.contrib.layers.l2_regularizer,
                             biases_initializer=tf.zeros_initializer(),
                             biases_regularizer=tf.contrib.layers.l2_regularizer
                             )
    
    return conv

def pool(conv):
    
    pool =  tf.contrib.layers.max_pool2d(conv, kernel_size=[2, 2],stride=2, padding='SAME')
    
    return pool


def convnet(image_arr, batch_size):
    
    conv1 = conv2d(image_arr, num_outputs=50, kernel_size=3)

    pool1 = pool(conv1) 
    
    conv2 = conv2d(pool1, num_outputs=50, kernel_size=3)
    
    pool2 = pool(conv2) 
    
    conv3 = conv2d(pool2, num_outputs=50, kernel_size=3)
    
    pool3 = pool(conv3) 
    
    conv4 = conv2d(pool3, num_outputs=50, kernel_size=3)
    
    pool4 = pool(conv4) 
    
    conv5 = conv2d(pool4, num_outputs=50, kernel_size=3)
    
    pool5 = pool(conv5)  
    
    conv6 = conv2d(pool5, num_outputs=50, kernel_size=3)
    
    pool6 = pool(conv6) 
    
    conv7 = conv2d(pool6, num_outputs=50, kernel_size=2)
    
    pool7 = pool(conv7)
    
    conv8 = conv2d(pool7, num_outputs = 1, kernel_size=1)
    
    final = tf.reshape(conv8, [batch_size, 1])
      
    return final

def loss(final, labels):
    
    log_loss = tf.losses.log_loss(labels, final)
    loss = tf.losses.add_loss(log_loss)
    
    total_loss = tf.losses.get_total_loss()
    
    
    return total_loss




    
    
     
    
                
     
                
     
                