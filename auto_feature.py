
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import scipy
import os
from scipy import misc
from scipy.misc import imread
from scipy.misc import imresize
from scipy import ndimage
import cv2
from skimage import color
from skimage import io
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt



## TRAIN DATA PREPROCESSING ##
image_dim=784
numImages=11741
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


for fname in os.listdir('/usr2/prouserdata/surya/TrackBali/glyph_train'):
	image=cv2.imread('/usr2/prouserdata/surya/TrackBali/glyph_train/'+fname,0)
        #img = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
        img_resize = cv2.resize(image,(28,28))
        img = cv2.bilateralFilter(img_resize,9,10,10)
        gamma_corrected = exposure.adjust_gamma(img, 2)
        imageflat=gamma_corrected.flatten() 
	train_set_img[i] = imageflat/255
	for k,c in enumerate(fname):
		if c=='_' :
			break
	str=fname[0:k]
	for m,string in enumerate(array):
		if str+'\r\n' == string:
			train_set_label[i][m]=1
		
	print(i)
	i=i+1

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






batch_size = 117
display_step = 10
examples_to_show = 6
total_batch = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer numfeatures
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="encoder_h1_weights"),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name= "encoder_h2_weights"),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1]), name="encoder_h1_biases"),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2]), name="encoder_h2_biases"),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2




# Construct model
encoder_op = encoder(X)




# Define loss and optimizer, minimize the squared error
# Initializing the variables
#init = tf.initialize_all_variables()

# Launch the graph

saver = tf.train.Saver()
feature_train = np.zeros((numImages, 128), dtype=np.float64)
feature_test = np.zeros((numImagesTest, 128), dtype=np.float64)


with tf.Session() as sess:
    k=0
    # Restore model
    saver.restore(sess, "model.ckpt")
    for i in range(total_batch):
        # Restore model
        batch_xs = train_set_img[k:k+batch_size]
        output = sess.run(encoder_op, feed_dict={X: batch_xs})
        for j in range(batch_size):
            feature_train[i*batch_size+j] = output[j]
     
        k = k + batch_size

    batch_size  = 100
    total_batch = 76
    k = 0

    for i in range(total_batch):
        batch_xs = test_set_img[k:k+batch_size]
        output = sess.run(encoder_op, feed_dict={X: batch_xs})
        for j in range(batch_size):
            feature_test[i*batch_size+j] = output[j]
     
        k = k + batch_size
# Softmax classifier on the features obtained by autoencoder (ie The middle compressed layer of activations when the image is passed in the net) 


## SOFTMAX CLASSIFIER ##

# Create the model
x = tf.placeholder(tf.float32, [None, 128])
W = tf.Variable(tf.zeros([128, 133]))
b = tf.Variable(tf.zeros([133]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 133])

 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  
# Train Step
train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


total_batch = 50
batch_size = 234
k = 0
for i in range(total_batch*200):
    if i%50 == 0:
        k=0
    batch_xs = feature_train[k:k+batch_size]
    batch_ys = train_set_label[k:k+batch_size]
    _, loss= sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys })
    print("Epoch:", '%04d' % (i+1),"cost=", "{:.9f}".format(loss)) 
    k = k+batch_size

test_accuracy = sess.run(accuracy, feed_dict = {x: feature_test, y_: test_set_label})
print("accuracy=", "{:.9f}".format(test_accuracy))




   
