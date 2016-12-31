# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
from optparse import OptionParser

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

parser = OptionParser()
parser.add_option("-l", "--learning_rate_initial", action="store", type="float", dest="learning_rate_initial", default=0.005)
parser.add_option("-m", "--multiplier", action="store", type="float", dest="multiplier", default=0.5)
parser.add_option("-n", "--num_steps", action="store", type="int", dest="num_steps", default=1001)
parser.add_option("-c", "--regularizationweight", action="store", type="float", dest="c", default=0)
parser.add_option("-w", "--weightpickle", action="store", type="string", dest="weightpickle", default="weightpickle")

[options, args] = parser.parse_args()
learning_rate_initial = options.learning_rate_initial
multiplier = options.multiplier
num_steps = options.num_steps
weightpickle = options.weightpickle
c = options.c

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# Model.
def model(data,keep_prob):
  conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
  hidden = tf.nn.relu(conv + layer1_biases)
  hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
  hidden= tf.nn.dropout(hidden, keep_prob)
  conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
  hidden = tf.nn.relu(conv + layer2_biases)
  hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
  hidden= tf.nn.dropout(hidden, keep_prob)
  shape = hidden.get_shape().as_list()
  reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
  hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
  hidden= tf.nn.dropout(hidden, keep_prob)
  layer = tf.matmul(hidden, layer4_weights) + layer4_biases
  hidden = tf.nn.relu(layer)
  return tf.matmul(hidden, layer5_weights) + layer5_biases


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

batch_size = 16
patch_size = 5
depth = 16
keep_prob = 1.00
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, 120], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[120]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [120, 84], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[84]))
  layer5_weights = tf.Variable(tf.truncated_normal(
      [84, num_labels], stddev=0.1))
  layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


  # Training computation.
  logits = model(tf_train_dataset,keep_prob)

  sum_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  regularization = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer1_biases)+
                   tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer2_biases)+
                   tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases)+
                   tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases)+
                   tf.nn.l2_loss(layer5_weights) + tf.nn.l2_loss(layer5_biases)
  ratio = tf.placeholder(tf.float32)
  loss = sum_loss+c*ratio*regularization
  # print(c)
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(learning_rate_initial, global_step,num_steps,multiplier)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

  # Optimizer.
  # optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  # valid_prediction = tf.matmul(tf_valid_dataset, weights[0]) + biases[0]
  # for j in range(number_of_hidden_layers):
  #   valid_prediction = tf.nn.relu(valid_prediction)
  #   valid_prediction = tf.matmul(valid_prediction, weights[j+1]) + biases[j+1]
  # valid_prediction = tf.nn.softmax(valid_prediction)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset,keep_prob))

  # test_prediction = tf.matmul(tf_test_dataset, weights[0]) + biases[0]
  # for j in range(number_of_hidden_layers):
  #   test_prediction = tf.nn.relu(test_prediction)
  #   test_prediction = tf.matmul(test_prediction, weights[j+1]) + biases[j+1]
  # test_prediction = tf.nn.softmax(test_prediction)
  test_prediction = tf.nn.softmax(model(tf_test_dataset,keep_prob))
  saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  previous_sum_loss = 0
  previous_regularization = 1
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, ratio: previous_sum_loss/previous_regularization}
    _, l, predictions, previous_sum_loss, previous_regularization, curratio = session.run(
      [optimizer, loss, train_prediction, sum_loss, regularization, ratio], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
      print('ratio ' + str(curratio))
      print('previous_sum_loss ' + str(previous_sum_loss))
      print('previous_regularization ' + str(previous_regularization))

  # Save the variables to disk.
  save_path = saver.save(session, weightpickle)
  print("Model saved in file: %s" % save_path)
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
