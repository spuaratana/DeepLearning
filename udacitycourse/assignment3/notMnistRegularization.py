# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

image_size = 28
num_labels = 10

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000
batch_size = 128
number_of_hidden_layers = 2
width_of_hidden_layer = 1024

graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = []
  biases = []
  keep_prob = tf.placeholder("float")

  weights.append(tf.Variable(tf.truncated_normal([image_size * image_size, width_of_hidden_layer])))
  biases.append(tf.Variable(tf.zeros([width_of_hidden_layer])))

  for j in range(number_of_hidden_layers-1):
    weights.append(tf.Variable(tf.truncated_normal([width_of_hidden_layer, width_of_hidden_layer])))
    biases.append(tf.Variable(tf.zeros([width_of_hidden_layer])))

  if(number_of_hidden_layers > 0):
    weights.append(tf.Variable(tf.truncated_normal([width_of_hidden_layer, num_labels])))
    biases.append(tf.Variable(tf.zeros([num_labels])))

  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  logits = tf.matmul(tf_train_dataset, weights[0]) + biases[0]
  for j in range(number_of_hidden_layers):
    logits = tf.nn.relu(logits)
    logits= tf.nn.dropout(logits, keep_prob)
    logits = tf.matmul(logits, weights[j+1]) + biases[j+1]

  c = 0.001
  sum_loss = tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
  for j in range(number_of_hidden_layers+1):
    sum_loss = sum_loss + c*tf.nn.l2_loss(weights[j]) + c*tf.nn.l2_loss(biases[j])

  loss = tf.reduce_mean(sum_loss)

  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.003, global_step,4501,0.5)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.matmul(tf_valid_dataset, weights[0]) + biases[0]
  for j in range(number_of_hidden_layers):
    valid_prediction = tf.nn.relu(valid_prediction)
    valid_prediction = tf.matmul(valid_prediction, weights[j+1]) + biases[j+1]
  valid_prediction = tf.nn.softmax(valid_prediction)

  test_prediction = tf.matmul(tf_test_dataset, weights[0]) + biases[0]
  for j in range(number_of_hidden_layers):
    test_prediction = tf.nn.relu(test_prediction)
    test_prediction = tf.matmul(test_prediction, weights[j+1]) + biases[j+1]
  test_prediction = tf.nn.softmax(test_prediction)

num_steps = 9001

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases.
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    # modstp = step%3 offset = modstp*batch_size
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 0.75}
    _, l, predictions, current_learning_rate = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Learning Rate ",current_learning_rate)
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
