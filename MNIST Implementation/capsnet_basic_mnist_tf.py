############# CAPSULE NETWORKS FOR MNIST ################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from __future__ import division, print_function, unicode_literals

#Import MNIST Dataset 
from tensorflow.examples.tutorials.mnist import input_data
dataset = input_data.read_data_sets('/temp/data')

#Image dims = 28 x 28
#First dim = None to be defined later as batch size; n_channels = 1 because Greyscale image
#Create a placeholder for the input
X = tf.placeholder(shape=[None, 28,28,1], dtype=tf.float32, name = 'X')

#Primary Capsules
num_maps_caps1 = 32
dimensions_caps1 = 8
num_capsules_caps1 = num_maps_caps1 * 6 * 6

#Apply Conv layers to compute the output
conv1 = tf.layers.conv2d(X, filters=256,kernel_size = 9, strides= 1, padding='valid', activation=tf.nn.relu, name = 'conv1')
conv2 = tf.layers.conv2d(conv1,filters=256, kernel_size=9, strides=2, padding='valid', activation=tf.nn.relu, name= 'conv2') #Shape = [batch_size, 6, 6, 256]
#Convert to [batch_size, 6x6x32, 8] because we need vectors of 8 dimensions
reshaped_caps1 = tf.reshape(conv2, [-1,num_capsules_caps1, dimensions_caps1], name = 'reshaped_caps1')

def squash_function(s,axis = -1, add_val = 1e-9, name = None):
    with tf.name_scope(name=name, default_name='squash_function'):
        s_squared = tf.reduce_sum(tf.square(s), axis = axis, keep_dims=True)
        s_add_val = tf.sqrt(s_squared + add_val)
        squash_factor = (s_squared/1. + s_squared)
        unit_vector = s/s_add_val
        return squash_factor * unit_vector

caps1_output = squash_function(reshaped_caps1, name = 'caps1_output')

#Compute Predicted Output vectors for each primary capsule-digit capsule pair
#Compute Routing by Agreement
dimensions_caps2 = 16
num_capsules_caps2 = 10 #1 for each digit
#Compute the weight matrix
W_matrix = tf.random_normal(shape=(1,num_capsules_caps1, num_capsules_caps2, dimensions_caps2, dimensions_caps1), 
                            stddev = 0.01, dtype=tf.float32, name = 'W_matrix')
W = tf.Variable(W_matrix, name = 'W')
#Tile it for the whole batch
batch_size = tf.shape(X)[0]
W_whole_batch = tf.tile(W, [batch_size,1,1,1,1], name = 'W_whole_batch')

#Shape of caps1_output = [batch_size, 1152, 8]. Convert to [batch_size, 1152, 10, 8, 1]
caps1_output_exp = tf.expand_dims(caps1_output,-1, name = 'caps1_output_exp')
caps1_output_tile = tf.expand_dims(caps1_output_exp,2,name = 'caps1_ouput_tile')
caps1_output_tiled = tf.tile(caps1_output_tile,[1,1,num_capsules_caps2,1,1], name = 'caps1_output_tiled')

predictions_caps2 = tf.matmul(W_whole_batch, caps1_output_tiled, name = 'predictions_caps2')

#Routing by Agreement
raw_weights = tf.zeros([batch_size, num_capsules_caps1,num_capsules_caps2, 1,1], dtype = np.float32, name = 'raw_weights')
routing_weights = tf.nn.softmax(raw_weights, dim=2, name = 'routing_weights')
weighted_predictions = tf.multiply(routing_weights, predictions_caps2, name = 'weighted_predictions')
weighted_sum = tf.reduce_sum(weighted_predictions, axis = 1, keep_dims = True, name = 'weighted_sum')

caps2_output_round_1 = squash_function(weighted_sum, axis=-2,name="caps2_output_round_1")

#Agreement - Round2
caps2_output_round_1_tiled = tf.tile(caps2_output_round_1,[1, num_capsules_caps1,1,1,1], name = 'caps2_outpout_round_1_tiled')
agreement = tf.matmul(predictions_caps2, caps2_output_round_1_tiled, transpose_a=True, name = 'agreement')
#Update routing weights
raw_weights_updated = tf.add(raw_weights,agreement, name = 'raw_weights_updated')

routing_weights_round_2 = tf.nn.softmax(raw_weights_updated,dim=2,name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2, predictions_caps2,name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,axis=1, keep_dims=True,name="weighted_sum_round_2")
caps2_output_round_2 = squash_function(weighted_sum_round_2,axis=-2,name="caps2_output_round_2")

caps2_output = caps2_output_round_2

def normalisation(s, axis = -1, epsilon = 1e-8, keep_dims = False, name = None):
    with tf.name_scope(name, default_name='normalisation'):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)
    
y_prob = normalisation(caps2_output, axis = -2, name = 'y_prob')
#Select the instance with the highest probability
y_prob_max = tf.argmax(y_prob, axis = 2, name = 'y_prob')
y_pred = tf.squeeze(y_prob_max, axis = [1,2], name = 'y_pred')


#Create Placeholder for labels
y = tf.placeholder(shape=[None], dtype=tf.int64, name = 'y')


#Calculate Margin Loss
T = tf.one_hot(y, depth=num_capsules_caps2, name = 'T')
caps2_output_norm = normalisation(caps2_output, axis = -2, keep_dims=True, name='caps2_output_norm')
present_error_raw = tf.square(tf.maximum(0.,0.9 - caps2_output_norm), name= 'present_error_raw')
present_error  =tf.reshape(present_error_raw, shape=(-1, 10),name="present_error")
absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - 0.1), name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 10), name="absent_error")
L = tf.add(T * present_error, 0.5 * (1.0 - T) * absent_error,name="L")
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")


#Apply masking
masks_labels = tf.placeholder_with_default(False,shape=(), name = 'masks_labels')
reconstruction_targets = tf.cond(masks_labels,
                                 lambda:y,
                                 lambda:y_pred,
                                 name = 'reconstruction_targets')
reconstruction_mask = tf.one_hot(reconstruction_targets, depth = num_capsules_caps2, name = 'reconstruction_mask')
#In order to multiply with caps2_output
reconstruction_mask = tf.reshape(reconstruction_mask, [-1,1, num_capsules_caps2,1,1], name = 'reconstruction_mask_reshaped')
caps2_output_masked = tf.multiply(caps2_output, reconstruction_mask, name = 'caps2_output_maksed')
#Flatten
decoder_input = tf.reshape(caps2_output_masked, [-1, num_capsules_caps2 * dimensions_caps2], name = 'decoder_input')


n_hidden1 = 512
n_hidden2 = 1024
n_output = 28 * 28

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,activation=tf.nn.relu,name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,activation=tf.nn.relu,name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,activation=tf.nn.sigmoid,name="decoder_output")
    
#Reconstruction_Loss - squared diff between i/p image and reconstructed image
X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output, name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference, name="reconstruction_loss")

#total loss
loss = tf.add(margin_loss, 0.0005 * reconstruction_loss, name = 'loss')

#Accuracy
correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

#Optimizer
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")

init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 10
batch_size = 50
restore_checkpoint = True

n_iterations_per_epoch = dataset.train.num_examples // batch_size
n_iterations_validation = dataset.validation.num_examples // batch_size
best_loss_val = np.infty
checkpoint_path = "./my_capsule_network"

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for epoch in range(n_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = dataset.train.next_batch(batch_size)
            # Run the training operation and measure the loss:
            _, loss_train = sess.run(
                [training_op, loss],
                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),y: y_batch,masks_labels: True})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(iteration, n_iterations_per_epoch,iteration * 100 / n_iterations_per_epoch,loss_train),end="")

        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch = dataset.validation.next_batch(batch_size)
            loss_val, acc_val = sess.run([loss, accuracy],feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iteration, n_iterations_validation,iteration * 100 / n_iterations_validation),end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(epoch + 1, acc_val * 100, loss_val," (improved)" if loss_val < best_loss_val else ""))

        # And save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val

#Evaluation
n_iterations_test = dataset.test.num_examples // batch_size

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    loss_tests = []
    acc_tests = []
    for iteration in range(1, n_iterations_test + 1):
        X_batch, y_batch = dataset.test.next_batch(batch_size)
        loss_test, acc_test = sess.run([loss, accuracy],feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),y: y_batch})
        loss_tests.append(loss_test)
        acc_tests.append(acc_test)
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iteration, n_iterations_test,iteration * 100 / n_iterations_test),end=" " * 10)
    loss_test = np.mean(loss_tests)
    acc_test = np.mean(acc_tests)
    print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
        acc_test * 100, loss_test))
    
    
#Predictions
n_samples = 5

sample_images = dataset.test.images[:n_samples].reshape([-1, 28, 28, 1])

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    caps2_output_value, decoder_output_value, y_pred_value = sess.run([caps2_output, decoder_output, y_pred],feed_dict={X: sample_images,y: np.array([], dtype=np.int64)})

sample_images = sample_images.reshape(-1, 28, 28)
reconstructions = decoder_output_value.reshape([-1, 28, 28])

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.imshow(sample_images[index], cmap="binary")
    plt.title("Label:" + str(mnist.test.labels[index]))
    plt.axis("off")

plt.show()

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.title("Predicted:" + str(y_pred_value[index]))
    plt.imshow(reconstructions[index], cmap="binary")
    plt.axis("off")
    
plt.show()
