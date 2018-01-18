import tensorflow as tf
import read
import convnet
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', help='GCS or local path to training data', required=True)
parser.add_argument('--output_dir', help='GCS or local path to output log', required=True)
parser.add_argument('--job-dir', help='GCS or local path job dir', default='junk')

args = parser.parse_args()
arguments = args.__dict__
arguments.pop('job-dir', None)

data_dir = arguments['data_dir']
output_dir = arguments['output_dir']


with tf.Graph().as_default():

    image_data, label_data = read.preprocess(filename = os.path.join(data_dir, 'train.json'))
    batch = read.batch(images = image_data, labels  = label_data)
#training  
  
    X = tf.placeholder(dtype = tf.float32, shape=(100, 75, 75, 2))
    Y = tf.placeholder(dtype = tf.int32, shape=(100, 1))
    batch_size = tf.placeholder(dtype = tf.int32)
        
    output = convnet.convnet(X, batch_size)
    
    loss = convnet.loss(output, Y)
    loss_summary = tf.summary.scalar(name = 'loss_summary', tensor = loss) 
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001)
    
    train_op = optimizer.minimize(loss)
    
    saver = tf.train.Saver()
    
#evaluation
    
    eval_images, eval_labels = batch.batches(mode= 'eval')
    eval_image_tensor = tf.placeholder(dtype = tf.float32, shape=(200, 75, 75, 2))
    eval_label_tensor = tf.placeholder(dtype = tf.int32, shape=(200, 1))
    
    eval_pred = tf.cast(tf.round(convnet.convnet(eval_image_tensor, batch_size = 200)), dtype = tf.int32)
    accuracy = tf.contrib.metrics.accuracy(eval_pred, eval_label_tensor)
    
    accuracy_summary = tf.summary.scalar(name = 'accuracy_summary', tensor = accuracy)
    
    summary_ops = tf.summary.merge_all()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        writer = tf.summary.FileWriter(output_dir, graph=tf.get_default_graph())
        
        for i in range(50000):
            
            images, labels = batch.batches(mode='train', step = i)
            
            sess.run(train_op, {X : images, Y : labels, batch_size : 100})
            
            if i % 500 == 0:
                
                saver.save(sess,output_dir, global_step = i)
                
            if i % 50 == 1:
               cost, acc, summary = sess.run([loss, accuracy, summary_ops], {X : images, Y : labels, 
                                             eval_image_tensor : eval_images, eval_label_tensor : eval_labels}) 
                                            
               writer.add_summary(summary, global_step = i)
               print(cost, acc)