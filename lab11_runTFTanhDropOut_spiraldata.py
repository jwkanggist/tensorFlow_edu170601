#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
#------------------------------------------------------------
    filename: lab6_runTFMultiANN_spiraldata.py

    A Multi-Hidden Layers Fully Connected Neural Network implementation with TensorFlow.
    This example is using two class spiral data

    written by Jaewook Kang @ Sep 2017
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

# file setup for tensorboard record
from datetime import datetime
now  = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs'
logdir="{}/run-{}/".format(root_logdir,now) #
#----------------

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

# #data plot
hfig1= plt.figure(1,figsize=[10,10])
plt.scatter(data.xdata1.values[0:int(data.xdata1.size/2)],\
            data.xdata2.values[0:int(data.xdata1.size/2)], \
            color='b',label='class0')
plt.scatter(data.xdata1.values[int(data.xdata1.size/2)+2:-1],\
            data.xdata2.values[int(data.xdata1.size/2)+2:-1], \
            color='r',label='class1')
plt.title('Two Spiral data Example')
plt.legend()


# configure training parameters =====================================
learning_rate = 0.01
training_epochs = 500
batch_size = 50
display_step = 1


# computational TF graph construction ================================
# Network Parameters
n_hidden_1 = 20 # 1st layer number of neurons
n_hidden_2 = 16 # 2nd layer number of neurons
n_hidden_3 = 8 # 3rd layer number of neurons
num_input = xsize   # two-dimensional input X = [x1 x2]
num_classes = ysize # 2 class

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
drop_prob= tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable (tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 7 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    drop_out1= tf.layers.dropout(layer_1,drop_prob)

    # Hidden fully connected layer with 7 neurons
    layer_2 = tf.add(tf.matmul(drop_out1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)
    drop_out2 = tf.layers.dropout(layer_2, drop_prob)

    # Hidden fully connected layer with 4 neurons
    layer_3 = tf.add(tf.matmul(drop_out2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.tanh(layer_3)
    drop_out3 = tf.layers.dropout(layer_3, drop_prob)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(drop_out3, weights['out']) + biases['out']
    return out_layer

# Construct model
#graph0 = tf.Graph()

#with graph0.as_default():
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

errRateTraining     = np.zeros(training_epochs)
errRateValidation   = np.zeros(training_epochs)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# summary for Tensorborard visualization
accuracy_summary    = tf.summary.scalar('accuracy',accuracy)
loss_summary        = tf.summary.scalar('loss',cost)

# file writing for Tensorboard
file_writer         = tf.summary.FileWriter(logdir=logdir)
file_writer.add_graph(tf.get_default_graph())

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(training_size/batch_size)
        # recording with tensorboard
        # 1) accuracy visualization
        summary_str = loss_summary.eval(feed_dict={X: x_training_data, Y: t_training_data, drop_prob: 0})
        file_writer.add_summary(summary_str,epoch)

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
                                                          Y: batch_ts, drop_prob: 0.2})

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

            errRateTraining[epoch] = 1.0 - accuracy.eval({X: batch_train_xs, \
                                                          Y: batch_train_ys, drop_prob: 0}, session=sess)

            errRateValidation[epoch] = 1.0 - accuracy.eval({X: batch_valid_xs, \
                                                            Y: batch_valid_ys, drop_prob: 0}, session=sess)

            print("Training set Err rate: %s"   % errRateTraining[epoch])
            print("Validation set Err rate: %s" % errRateValidation[epoch])

        print("--------------------------------------------")

    print("Optimization Finished!")

    # Calculate accuracy for test images
    ##-------------------------------------------
    # # training Result display
    print("Validation set Err rate:", accuracy.eval({X: x_validation_data, Y: t_validation_data, drop_prob: 0},session=sess)/validation_size)


hfig2 = plt.figure(2,figsize=(10,10))
epoch_index = np.array([elem for elem in range(training_epochs)])
plt.semilogy(epoch_index,errRateTraining,label='Training data',color='r',marker='o')
plt.semilogy(epoch_index,errRateValidation,label='Validation data',color='b',marker='x')

plt.legend()
plt.title('Classification Error Rate of prediction:')
plt.xlabel('Iteration epoch')
plt.ylabel('error Rate')

plt.show()

# close tensorboard recording
file_writer.close()