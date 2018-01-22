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
    tf.set_random_seed(1234)

    image_data, label_data = read.preprocess(file = os.path.join(data_dir, 'train.json'))
    batch = read.batch(images = image_data, labels  = label_data)
#training  
  
    X = tf.placeholder(dtype = tf.float32, shape=(100, 75, 75, 2))
    Y = tf.placeholder(dtype = tf.int32, shape=(100, 1))
    
        
    output = convnet.convnet(X)
    
    loss = convnet.loss(output, Y)
    loss_summary = tf.summary.scalar(name = 'loss_summary', tensor = loss) 
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    
    grads = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads)
    
    saver = tf.train.Saver(save_relative_paths= True)
 
#eval_ops    
    eval_images = tf.placeholder(dtype = tf.float32, name= 'eval_images')
    eval_op = convnet.convnet(eval_images) 
    
#summary ops
    for index, grad in enumerate(grads):
        
        tf.summary.tensor_summary("{}-grad".format(grads[index][1].name), grads[index])
        
    summary_ops = tf.summary.merge_all()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        writer = tf.summary.FileWriter(output_dir, graph=tf.get_default_graph())
        
        for i in range(10000):
            
            images, labels = batch.batches(mode='train', step = i)
            
            sess.run(train_op, {X : images, Y : labels})
            
            if i % 500 == 0:
                
                saver.save(sess,output_dir, global_step = i, write_meta_graph= False)
                
            if i % 50 == 1:
               cost, summary = sess.run([loss,summary_ops], {X : images, Y : labels}) 
                                        
                                            
               writer.add_summary(summary, global_step = i)
               print(cost)