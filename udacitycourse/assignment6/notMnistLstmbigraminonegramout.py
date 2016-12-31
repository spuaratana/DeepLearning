# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-l", "--learning_rate_initial", action="store", type="float", dest="learning_rate_initial", default=10.0)
parser.add_option("-m", "--multiplier", action="store", type="float", dest="multiplier", default=0.1)
parser.add_option("-d", "--decay_steps", action="store", type="int", dest="decay_steps", default=5001)
parser.add_option("-n", "--num_steps", action="store", type="int", dest="num_steps", default=10001)
parser.add_option("-w", "--weightpickle", action="store", type="string", dest="weightpickle", default="weightpickle")

[options, args] = parser.parse_args()
learning_rate_initial = options.learning_rate_initial
multiplier = options.multiplier
num_steps = options.num_steps
decay_steps = options.decay_steps
weightpickle = options.weightpickle

class BigramBatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // (batch_size)
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()

  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    vocabulary_size = len(string.ascii_lowercase) + 1
    batch = np.zeros(shape=(self._batch_size, vocabulary_size*vocabulary_size), dtype=np.float)
    for b in range(self._batch_size):
      bigram = self._text[self._cursor[b]] + self._text[self._cursor[b]+1]
      bigram_id = char2id(self._text[self._cursor[b]])*vocabulary_size + char2id(self._text[self._cursor[b]+1])
    #   print(bigram_id)
      batch[b,bigram_id] = 1.0
      self._cursor[b] = (self._cursor[b] + 2) % self._text_size
    return batch

  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batch = self._next_batch()
      batches.append(batch)
    self._last_batch = batches[-1]
    return batches

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

def read_data(filename):
  with zipfile.ZipFile(filename) as f:
    name = f.namelist()[0]
    data = tf.compat.as_str(f.read(name))
  return data

def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0

def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def onehotbigram(bigram_id):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size*vocabulary_size], dtype=np.float)
  p[0, bigram_id] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]

def firstcharelementlist(batches):
    firstbatches = []
    for i in range(len(batches)):
        firstbatches.append(firstcharelement(batches[i]))
    return firstbatches

def firstcharelement(batch):
    firstcharbatch = np.zeros(shape=(batch.shape[0], vocabulary_size), dtype=np.float)
    for j in range(batch.shape[0]):
        indices = np.argmax(batch[j,:])
        firstindices = indices//vocabulary_size
        firstcharbatch[j,firstindices]=1.0
    return firstcharbatch

def build_bigram_dictionary():
    numchar = len(string.ascii_lowercase) + 1
    dictionary = dict()
    for i in range(numchar):
        for j in range(numchar):
            firstchar = id2char(i)
            secondchar = id2char(j)
            bigram = firstchar + secondchar
            dictionary[bigram] = i*numchar + j
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def random_bigram():
  bigram = np.random.randint(0, vocabulary_size*vocabulary_size-1, size=[1, 1])
  return bigram[0,0]

url = 'http://mattmahoney.net/dc/'
filename = maybe_download('text8.zip', 31344016)
text = read_data(filename)
print('Data size %d' % len(text))
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
embedding_size = 128
first_letter = ord(string.ascii_lowercase[0])
batch_size=64
num_unrollings=10
dropout_keepprob = 1.0
train = BigramBatchGenerator(train_text, batch_size, num_unrollings)
valid = BigramBatchGenerator(valid_text, 1, 1)

num_nodes = 64

graph = tf.Graph()
with graph.as_default():

  # Parameters:
  embeddings = tf.Variable(tf.truncated_normal([vocabulary_size*vocabulary_size,embedding_size], -0.1, 0.1), trainable=False)
  pickle_file = 'bigramembeddings.pickle'

  try:
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      embeddings_initial = save['embeddings']
      embeddings.assign(embeddings_initial)
  except:
      print("Initial embeddings not found\n")
  # embeddings.assign(embeddings_initial)
  # Input gate: input, previous output, and bias.
  ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ib = tf.Variable(tf.zeros([1, num_nodes]))
  # Forget gate: input, previous output, and bias.
  fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  fb = tf.Variable(tf.zeros([1, num_nodes]))
  # Memory cell: input, state and bias.
  cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  cb = tf.Variable(tf.zeros([1, num_nodes]))
  # Output gate: input, previous output, and bias.
  ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ob = tf.Variable(tf.zeros([1, num_nodes]))
  #Concatenate weights
  ax = tf.concat(1,[ix,fx,cx,ox])
  am = tf.concat(1,[im,fm,cm,om])
  ab = tf.concat(1,[ib,fb,cb,ob])
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))

  dictionary, reverse_dictionary = build_bigram_dictionary()
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""
    amatmul = tf.matmul(i, ax) + tf.matmul(o, am) + ab
    amatmul_input,amatmul_forget,amatmul_update,amatmul_output = tf.split(1,4,amatmul)
    input_gate = tf.sigmoid(amatmul_input)
    forget_gate = tf.sigmoid(amatmul_forget)
    update = amatmul_update
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(amatmul_output)
    return output_gate * tf.tanh(state), state

  # Input data.
  train_data = list()
  train_label = list()
  for _ in range(num_unrollings):
    train_data.append(
      tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size*vocabulary_size]))
    train_label.append(
      tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  train_inputs = train_data[:]
  train_labels = train_label[:]  # labels are inputs shifted by one time step.

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    embedding = tf.matmul(i, embeddings)
    output, state = lstm_cell(embedding, output, state)
    outputs.append(output)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.nn.dropout(tf.concat(0,outputs),dropout_keepprob), w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        logits, tf.concat(0,train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    learning_rate_initial, global_step, decay_steps, multiplier, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)

  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size*vocabulary_size])
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  embedding = tf.matmul(sample_input, embeddings)
  sample_output, sample_state = lstm_cell(
    embedding, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

summary_frequency = 100
# saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train.next()
    batches_labels = firstcharelementlist(batches)
    feed_dict = dict()
    for i in range(num_unrollings):
      feed_dict[train_data[i]] = batches[i]
      feed_dict[train_label[i]] = batches_labels[i+1]
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print(
        'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches_labels)[1:])
      print('Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions,labels))))
      if step % (summary_frequency) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          bigram_id = random_bigram()
          bigram_char = reverse_dictionary[bigram_id]
          sentence = bigram_char
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input:onehotbigram(bigram_id)})
            output_id = np.argmax(sample(prediction), 1)
            output_id = output_id[0]
            bigram_char = bigram_char[-1]+id2char(output_id)
            sentence += id2char(output_id)
            bigram_id = dictionary[bigram_char]
          print(sentence)
        print('=' * 80)
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size):
        b = valid.next()
        bl = firstcharelementlist(b)
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, bl[1])
      print('Validation set perplexity: %.2f' % float(np.exp(
        valid_logprob / valid_size)))
  # save_path = saver.save(session, weightpickle)
  # print("Model saved in file: %s" % save_path)
