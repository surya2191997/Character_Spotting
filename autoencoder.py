# Autoencoder for extracting features


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
import math



## TRAIN DATA PREPROCESSING ##
image_dim=784
numImages=11741
num_of_classes=133
i=0
j=0
train_set_img = np.zeros((numImages, image_dim), dtype=np.float64)
train_set_label = np.zeros((numImages,num_of_classes), dtype=np.float64)





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



# Parameters
learning_rate = 0.005
training_epochs = 1000
batch_size = 117
display_step = 10
examples_to_show = 6

# Network Parameters
n_hidden_1 = 256 # 1st layer numfeatures
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="encoder_h1_weights"),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="encoder_h2_weights"),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]), name="decoder_h1_weights"),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]), name="decoder_h2_weigths"),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1]), name="encoder_h1_biases"),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2]), name="encoder_h2_biases"),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1]), name="decoder_h1_biases"),
    'decoder_b2': tf.Variable(tf.random_normal([n_input]), name="decoder_h2_biases"),
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


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

#Add saver op
saver = tf.train.Saver()
array =  np.zeros((n_input,n_hidden_1), dtype=np.float64)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = 100
    k=0
	
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        k=0
        #np.random.shuffle(train_set_img)
        for i in range(total_batch):
            batch_xs = train_set_img[k:k+batch_size]
            # Run optimization op (backprop) and costop (to get loss value)
            _, c,array = sess.run([optimizer, cost,weights['encoder_h1']], feed_dict={X: batch_xs})
            k=k+batch_size
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X:train_set_img[:examples_to_show]})
    # Compare original images with their reconstructions
    #f, a = plt.subplots(2, 10, figsize=(10, 2))
    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in file: %s" % save_path)

s1=['original1.png','original2.png','original3.png','original4.png','original5.png','original6.png']
s2=['pro1.png','pro2.png','pro3.png','pro4.png','pro5.png','pro6.png']

for i in range(examples_to_show):
	scipy.misc.imsave(s1[i],np.reshape(train_set_img[i], (28, 28)))
	scipy.misc.imsave(s2[i],np.reshape(encode_decode[i], (28, 28)))

second_layer =30
f, a = plt.subplots(5, 6, figsize=(6, 5),squeeze = False)
#print(weights['encoder_h1'][0][0])

for i in range(second_layer):
    array1 =  np.zeros(image_dim, dtype=np.float64)
    for j in range(n_input):
        array1[j] = array[j][i]
    
    sum = 0
    for j in range(n_input):
        sum=sum+array1[j]*array1[j]
    sqrt = sum**(0.5)
    array1 = array1/sqrt
    k1 = int(i/6)
    k2 = int(i%6) 
    a[k1][k2].imshow(np.reshape(array1, (28, 28)))
    
    f.show()
    plt.draw()
    plt.waitforbuttonpress()

