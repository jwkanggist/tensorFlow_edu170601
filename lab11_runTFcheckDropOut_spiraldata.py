#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
#------------------------------------------------------------
    filename: lab11_runTFcheckDropOut_spiraldata.py

    To check effect of dropout in DNN network

    This example data set is using two class spiral data

    written by Jaewook Kang @ Jan 2018
#------------------------------------------------------------
'''


from os import getcwd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_io


# reading data set from csv file ==========================
xsize  = 2
ysize  = 2

data = pd.read_csv('./data/twospirals_N5000.csv')
data.columns=['xdata1','xdata2','tdata']
permutation_index = np.random.permutation(data.index)
permutated_data = data.reindex(permutation_index)
permutated_data.columns=['xdata1','xdata2','tdata']

x_data = np.zeros([permutated_data.xdata1.size,xsize])
x_data[:,0] = permutated_data.xdata1.values
x_data[:,1] = permutated_data.xdata2.values

t_data = np.zeros([permutated_data.tdata.size,ysize])
t_data[:,0] = permutated_data.tdata.values
t_data[:,1] = np.invert(permutated_data.tdata.values) + 2


total_size = permutated_data.xdata1.size
training_size = int(np.floor(permutated_data.xdata1.size * 0.8))
validation_size = total_size - training_size



# data dividing
x_training_data = x_data[0:training_size,:]
t_training_data = t_data[0:training_size,:]

x_validation_data = x_data[training_size:-1,:]
t_validation_data = t_data[training_size:-1,:]



# configure training parameters =====================================
learning_rate = 0.005

training_epochs = 5000
batch_size = 500
display_step = 100
total_batch = int(training_size / batch_size)

# dropout_rate = 1 --> no dropout
# dropout_rate = 1 --> no nodes to work
dropoutrate_in_training = 0.8

# computational TF graph construction ================================
# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 7 # 2nd layer number of neurons
n_hidden_3 = 7 # 3rd layer number of neurons
n_hidden_4 = 4 # 4rd layer number of neurons
n_hidden_5 = 4 # 5rd layer number of neurons


num_input = xsize   # two-dimensional input X = [x1 x2]
num_classes = ysize # 2 class

# tf Graph input
X           = tf.placeholder(tf.float32, [None, num_input])
Y           = tf.placeholder(tf.float32, [None, num_classes])

# droprate must be given by placeholder 
# since the network in validation do not require dropout nodes in layers.
dropoutrate_io   = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input,  n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'out':tf.Variable(tf.random_normal([n_hidden_5, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    # Input fully connected layer with 10 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.layers.dropout(inputs=layer_1,rate=dropoutrate_in_training)


    # Hidden fully connected layer with 7 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.layers.dropout(inputs=layer_2,rate=  dropoutrate_in_training)


    # Hidden fully connected layer with 7 neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.layers.dropout(inputs=layer_3,rate= dropoutrate_in_training)

    # Hidden fully connected layer with 4 neurons
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    layer_4 = tf.layers.dropout(inputs=layer_4,rate = dropoutrate_in_training)

    # Hidden fully connected layer with 4 neurons
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.relu(layer_5)
    layer_5 = tf.layers.dropout(inputs=layer_5, rate=dropoutrate_in_training)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

## when you use AdamOptimizer, instead of SGD, the error rate immediately becomes near zero.
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

errRatebyTrainingSet     = np.zeros(training_epochs)
errRatebyValidationSet   = np.zeros(training_epochs)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# for visualization of vanishing gradient problem
grad_wrt_weight_layer1_tensor = tf.gradients(cost,weights['h1'],\
                                             name='grad_wrt_weight_layer1')
grad_wrt_weight_layer2_tensor = tf.gradients(cost,weights['h2'],\
                                             name='grad_wrt_weight_layer2')
grad_wrt_weight_layer3_tensor = tf.gradients(cost,weights['h3'],\
                                             name='grad_wrt_weight_layer3')
grad_wrt_weight_layer4_tensor = tf.gradients(cost,weights['h4'],\
                                             name='grad_wrt_weight_layer4')
grad_wrt_weight_layer5_tensor = tf.gradients(cost,weights['h5'],\
                                             name='grad_wrt_weight_layer5')

grad_wrt_weight_layer1_iter = np.zeros([total_batch,1])
grad_wrt_weight_layer2_iter = np.zeros([total_batch,1])
grad_wrt_weight_layer3_iter = np.zeros([total_batch,1])
grad_wrt_weight_layer4_iter = np.zeros([total_batch,1])
grad_wrt_weight_layer5_iter = np.zeros([total_batch,1])


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.

        weight_array = list()
        for i in range(total_batch):
            data_start_index = i * batch_size
            data_end_index = (i + 1) * batch_size
            # feed traing data --------------------------
            batch_xs = x_training_data[data_start_index:data_end_index, :]
            batch_ts = t_training_data[data_start_index:data_end_index, :]

            #----------------------------------------------
            # Run optimization op (backprop) and cost op (to get loss value)
            # feedign training data
            _, local_batch_cost = sess.run([optimizer,cost], feed_dict={X: batch_xs,
                                                          Y: batch_ts})

            if epoch == training_epochs - 1:
                # print ('Gradient calculation to see gradient vanishing problem')
                _, grad_wrt_weight_layer1 = sess.run([optimizer,grad_wrt_weight_layer1_tensor], feed_dict={X: batch_xs,
                                                          Y: batch_ts, dropoutrate_io: dropoutrate_in_training})
                _, grad_wrt_weight_layer2 = sess.run([optimizer,grad_wrt_weight_layer2_tensor], feed_dict={X: batch_xs,
                                                          Y: batch_ts, dropoutrate_io: dropoutrate_in_training})
                _, grad_wrt_weight_layer3 = sess.run([optimizer,grad_wrt_weight_layer3_tensor], feed_dict={X: batch_xs,
                                                          Y: batch_ts, dropoutrate_io: dropoutrate_in_training})
                _, grad_wrt_weight_layer4 = sess.run([optimizer,grad_wrt_weight_layer4_tensor], feed_dict={X: batch_xs,
                                                          Y: batch_ts, dropoutrate_io: dropoutrate_in_training})
                _, grad_wrt_weight_layer5 = sess.run([optimizer,grad_wrt_weight_layer5_tensor], feed_dict={X: batch_xs,
                                                          Y: batch_ts, dropoutrate_io: dropoutrate_in_training})
                grad_wrt_weight_layer1 = np.array(grad_wrt_weight_layer1)
                grad_wrt_weight_layer2 = np.array(grad_wrt_weight_layer2)
                grad_wrt_weight_layer3 = np.array(grad_wrt_weight_layer3)
                grad_wrt_weight_layer4 = np.array(grad_wrt_weight_layer4)
                grad_wrt_weight_layer5 = np.array(grad_wrt_weight_layer5)

                grad_wrt_weight_layer1 = grad_wrt_weight_layer1.reshape(grad_wrt_weight_layer1.shape[1],
                                                                    grad_wrt_weight_layer1.shape[2])
                grad_wrt_weight_layer2 = grad_wrt_weight_layer2.reshape(grad_wrt_weight_layer2.shape[1],
                                                                    grad_wrt_weight_layer2.shape[2])
                grad_wrt_weight_layer3 = grad_wrt_weight_layer3.reshape(grad_wrt_weight_layer3.shape[1],
                                                                    grad_wrt_weight_layer3.shape[2])
                grad_wrt_weight_layer4 = grad_wrt_weight_layer4.reshape(grad_wrt_weight_layer4.shape[1],
                                                                    grad_wrt_weight_layer4.shape[2])
                grad_wrt_weight_layer5 = grad_wrt_weight_layer5.reshape(grad_wrt_weight_layer5.shape[1],
                                                                    grad_wrt_weight_layer5.shape[2])

                grad_wrt_weight_layer1_iter[i] = grad_wrt_weight_layer1.mean()
                grad_wrt_weight_layer2_iter[i] = grad_wrt_weight_layer2.mean()
                grad_wrt_weight_layer3_iter[i] = grad_wrt_weight_layer3.mean()
                grad_wrt_weight_layer4_iter[i] = grad_wrt_weight_layer4.mean()
                grad_wrt_weight_layer5_iter[i] = grad_wrt_weight_layer5.mean()

            # Compute average loss
            avg_cost += local_batch_cost / total_batch

            # print ("At %d-th batch in %d-epoch, avg_cost = %f" % (i,epoch,avg_cost) )

            # Display logs per epoch step


        if display_step == 0:
            continue
        elif (epoch + 1) % display_step == 0:
            # print("Iteration:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            batch_train_xs = x_training_data
            batch_train_ys = t_training_data
            batch_valid_xs = x_validation_data
            batch_valid_ys = t_validation_data

            # for error rate evaluation, the dropout rate must be 1.0
            errRatebyTrainingSet[epoch] = 1.0 - accuracy.eval({X: batch_train_xs, \
                                                          Y: batch_train_ys, dropoutrate_io: 1.0}, session=sess)

            errRatebyValidationSet[epoch] = 1.0 - accuracy.eval({X: batch_valid_xs, \
                                                            Y: batch_valid_ys, dropoutrate_io: 1.0}, session=sess)

            print("Training set Err rate: %s"   % errRatebyTrainingSet[epoch])
            print("Validation set Err rate: %s" % errRatebyValidationSet[epoch])
            print("--------------------------------------------")

    print("Optimization Finished!")

    # Calculate accuracy for test images
    ##-------------------------------------------
#data plot
hfig1= plt.figure(1,figsize=[10,10])
plt.scatter(data.xdata1.values[0:int(data.xdata1.size/2)],\
            data.xdata2.values[0:int(data.xdata1.size/2)], \
            color='b',label='class0')
plt.scatter(data.xdata1.values[int(data.xdata1.size/2)+2:-1],\
            data.xdata2.values[int(data.xdata1.size/2)+2:-1], \
            color='r',label='class1')
plt.title('Two Spiral data Example')
plt.legend()


hfig2 = plt.figure(2,figsize=(10,10))
batch_index = np.array([elem for elem in range(total_batch)])
plt.plot(batch_index,grad_wrt_weight_layer1_iter,label='layer1',color='b',marker='o')
plt.plot(batch_index,grad_wrt_weight_layer4_iter,label='layer4',color='y',marker='o')
plt.plot(batch_index,grad_wrt_weight_layer5_iter,label='layer5',color='r',marker='o')
plt.legend()
plt.title('Dropout = (%s), Weight Gradient over minibatch iter' % dropoutrate_in_training)
plt.xlabel('minibatch iter')
plt.ylabel('Weight Gradient')

hfig3 = plt.figure(3,figsize=(10,10))
epoch_index = np.array([elem for elem in range(training_epochs)])
plt.plot(epoch_index,errRatebyTrainingSet,label='Training data',color='r',marker='o')
plt.plot(epoch_index,errRatebyValidationSet,label='Validation data',color='b',marker='x')
plt.legend()
plt.title('Dropout = (%s), Train/Valid Err' % dropoutrate_in_training)
plt.xlabel('Iteration epoch')
plt.ylabel('error Rate')



