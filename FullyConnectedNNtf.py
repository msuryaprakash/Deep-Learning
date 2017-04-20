import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#3 hidden layers
h_nodes_hl1 = 500
h_nodes_hl2 = 500
h_nodes_hl3 = 500

n_classes = 10
batch_size = 100

#Sometimes size is useful
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, h_nodes_hl1])),
					   'biases': tf.Variable(tf.random_normal([h_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([h_nodes_hl1, h_nodes_hl2])),
					   'biases': tf.Variable(tf.random_normal([h_nodes_hl2]))}
					
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([h_nodes_hl2, h_nodes_hl3])),
					   'biases': tf.Variable(tf.random_normal([h_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([h_nodes_hl3, n_classes])),
					   'biases': tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	return output


def train(x, y):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
	#learning_rate = 0.001 by default
	optimier = tf.train.AdamOptimizer().minimize(cost)

	epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimier, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch: ', i, ' completed out of ', epochs, ' Loss: ', epoch_loss)


		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy: ", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train(x, y)
