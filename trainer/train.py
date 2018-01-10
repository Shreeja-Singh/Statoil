import tensorflow as tf
import read
import convnet
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', help='GCS or local path to training data', required=True)
parser.add_argument('--output_dir', help='GCS or local path to output log', required=True)
parser.add_argument('--job_dir', help='GCS or local path job dir', default='junk')

args, unknown = parser.parse_known_args()
arguments = args.__dict__

data_dir = os.path.join(os.getcwd(), arguments['data_dir'])
output_dir = os.path.join(os.getcwd(), arguments['output_dir'])


with tf.Graph().as_default():

    image_data, label_data = read.preprocess(file = os.path.join(data_dir,'train.json'))

#training    
    X = tf.placeholder(dtype = tf.float64)
    Y = tf.placeholder(dtype = tf.float64)
    batch_size = tf.placeholder(dtype = tf.float64)
    
    output = convnet.convnet(X, batch_size)
    
    loss = convnet.loss(output, Y)
    loss_summary = tf.summary.scalar(name = 'loss_summary', tensor = loss) 
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
    
    train_op = optimizer.minimize(loss)
    
    saver = tf.train.Saver()
    
#evaluation

    eval_batch = read.batch(mode = 'eval', step = 0, images = image_data, labels  = label_data)
    eval_images, eval_labels = eval_batch.batches()
    
    eval_pred = convnet.convnet(eval_images, batch_size = 200)
    accuracy = tf.contrib.metrics.accuracy(tf.round(eval_pred), eval_labels)
    
    accuracy_summary = tf.summary.scalar(name = 'accuracy_summary', tensor = accuracy)
    
    summary_ops = tf.summary.merge_all()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        writer = tf.summary.FileWriter(output_dir, graph=tf.get_default_graph())
        
        for i in range(50000):
            
            batch = read.batch(mode = 'train', step = i, images = image_data, labels  = label_data)
            images, labels = batch.batches()
            sess.run(train_op, { 'X' : images, 'Y' : labels, 'batch_size' : 100})
            
            if i % 500 == 0:
                
                saver.save(sess,output_dir, global_step = i)
                
            if i % 50 == 1:
               cost, acc, summary = sess.run([loss, accuracy, summary_ops])
               writer.add_summary(summary, global_step = i)
               print(cost, acc)
               
                
                
        
        
        
     
