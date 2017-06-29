# Convolutional Neural Net , Architecture as given in paper



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import numpy as np
import os
import scipy
from scipy.misc import imread
from scipy.misc import imresize
from scipy import ndimage
import cv2
from skimage import color
from skimage import io
import cv2
from skimage import data, exposure, img_as_float


from tensorflow.contrib import learn
#from tensorflow.python import debug as tf_debug
import tensorflow.contrib.learn.python.learn.estimators

#sess = tf_debug.Local1CLIDebugWrapperSession(sess)
#sess.add_tensor_filter("has_inf_r_nan",tf_debug.has_inf_or_nan)



FLAGS = None

image_dim=784
numImages=11700
numImagesTest=7600
num_of_classes=133
i=0
j=0
train_set_img = np.zeros((numImages, image_dim), dtype = np.float64)
train_set_label = np.zeros((numImages,num_of_classes), dtype=np.float64)
test_set_img = np.zeros((numImagesTest, image_dim), dtype=np.float64)
test_set_label = np.zeros((numImagesTest,num_of_classes), dtype=np.float64)





with open("/usr2/prouserdata/surya/TrackBali/list_class_name.txt", "r") as ins:
    array = []
    for line in ins:
        array.append(line)

list = []
for fname in os.listdir('/usr2/prouserdata/surya/TrackBali/glyph_train'):
  image=cv2.imread('/usr2/prouserdata/surya/TrackBali/glyph_train/'+fname)
  img = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
  img_resize = cv2.resize(img,(28,28))
  img = cv2.bilateralFilter(img_resize,9,10,10)
  gamma_corrected = exposure.adjust_gamma(img, 2)
  imageflat=gamma_corrected.flatten() 
  train_set_img[i] = imageflat
  for k,c in enumerate(fname):
    if c=='_' :
      break
  str=fname[0:k]
  for m,string in enumerate(array):
    if str+'\r\n' == string:
      train_set_label[i][m]=1
    
  print(i) 
  i=i+1
  if i==11700:
    break

#train_set_img = np.array(list)



i=0
j=0
sum=0

for j in range(image_dim):
  sum = 0
  for i in range(numImages):
    sum=sum+train_set_img[i][j]

  for i in range(numImages):
    train_set_img[i][j] = train_set_img[i][j]-sum/numImages

  print('j: %d  mean: %g'% (j,sum/numImages))

i=0
j=0

for fname in os.listdir('/usr2/prouserdata/surya/TrackBali/glyph_test'):
  image=cv2.imread('/usr2/prouserdata/surya/TrackBali/glyph_test/'+fname)
  img = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
  img_resize = cv2.resize(img,(28,28))
  img = cv2.bilateralFilter(img_resize,9,10,10)
  gamma_corrected = exposure.adjust_gamma(img, 2)
  imageflat=gamma_corrected.flatten() 
  test_set_img[i] = imageflat
  for k,c in enumerate(fname):
    if c=='_' :
      break
  str=fname[0:k]
  for m,string in enumerate(array):
    if str+'\r\n' == string:
      test_set_label[i][m]=1
    
  print(i)
  i=i+1
  if i==7600:
    break



i=0
j=0
sum=0

for j in range(image_dim):
  sum=0
  for i in range(numImagesTest):
    sum=sum+test_set_img[i][j]

  for i in range(numImagesTest):
    test_set_img[i][j] = test_set_img[i][j]-sum/numImagesTest

  print('j: %d  mean: %g'% (j,sum/numImagesTest))


#test_set_img=np.array(list)


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)




def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 28, 28, 1])
	# First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	
  #tf.histogram_summary("weights1",W_conv1)
  #tf.histogram_summary("bias1",b_conv1)
  #tf.histogram_summary("activation1",h_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  
  #tf.histogram_summary("weights2",W_conv1)
  #tf.histogram_summary("bias2",b_conv2)
  #tf.histogram_summary("activation2",h_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([])

  
  


  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  
  #tf.histogram_summary("weights_fc1",W_fc1)
  #tf.histogram_summary("bias_fc1",b_fc1)
  #tf.histogram_summary("activation_fc1",h_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

   # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 133])
  b_fc2 = bias_variable([133])
  

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  regularizer = tf.nn.l2_loss(W_conv1)+tf.nn.l2_loss(W_conv2)+tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc2)
  
  #tf.histogram_summary("weights_fc2",W_fc2)
  #tf.histogram_summary("bias_fc2",b_fc2)
  #tf.histogram_summary("activation_fc2",y_conv)


  return y_conv, keep_prob, regularizer




def main(_):
  # Import data
  #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 133])

  # Build the graph for the deep net
  y_conv, keep_prob, regularizer = deepnn(x)

  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  
  
  loss =  loss + 0.01 * regularizer
  learning_rate = 2e-3

  train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  #tf.scalar_summary('loss',loss)
  #tf.scalar_summary('accuracy', accuracy)

  start = 0
  start_2=0
  end = 11700
  batch_size = 234
  batch_size_2 = 100  
  count = 0

  #logs_path = '/home/bt3/15CS10044'
  #merged_summary = tf.merge_all_summaries()
  #writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
  
  	
    for i in range(9000):
      
      if i%50 == 0:
        start = 0
        arr = np.zeros((numImages, image_dim+num_of_classes), dtype=np.float64)
        arr = np.concatenate((train_set_img,train_set_label),axis=1)
        np.random.shuffle(arr)
        for j in range(numImages):
          train_set_img[j]=arr[j][0:image_dim]
          train_set_label[j]=arr[j][image_dim:image_dim+num_of_classes]
    
      if i%76 == 0:
        start_2=0

      if i%1000 == 0:
        learning_rate=learning_rate/2
    
      batch_x = train_set_img[start:start+batch_size]
      batch_y = train_set_label[start:start+batch_size]
      
     
      #if i%5 ==0:
        #s = sess.run(merged_summary, feed_dict ={x: batch_x, y_: batch_y, keep_prob: 1.0})
        #writer.add_summary(s,i)

      train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
      test_accuracy = accuracy.eval(feed_dict={x: test_set_img[start_2:start_2+batch_size_2] , y_: test_set_label[start_2:start_2+batch_size_2] ,keep_prob: 1.0})
      loss_train = loss.eval(feed_dict={ x: batch_x, y_ : batch_y, keep_prob: 1.0})
      
   

      print('step %d, training accuracy %g, test accuracy %g, loss %g' % (i, train_accuracy,test_accuracy,loss_train))
      train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
      start = start+batch_size
      start_2 = start_2 + batch_size_2
      
    start = 0
    batch_size = 100
    n=0
    curr=0
    acc=0
    #print('test accuracy %g' % accuracy.eval(feed_dict={x: test_set_img, y_: test_set_label, keep_prob: 1.0}))
    for i in range(76):
      curr=accuracy.eval(feed_dict={x: test_set_img[start:start+batch_size], y_: test_set_label[start:start+batch_size], keep_prob: 1.0})
      acc=(n*acc+batch_size*curr)/(n+batch_size)
      print(acc)
      n=n+batch_size
      start=start+batch_size
    print('test accuracy %g' % acc)
    


if __name__ == '__main__':
  tf.app.run()
