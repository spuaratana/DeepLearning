# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import string
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sklearn.manifold import TSNE

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

def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

def read_data(filename):
  with zipfile.ZipFile(filename) as f:
    name = f.namelist()[0]
    data = tf.compat.as_str(f.read(name))
  return data

def id2char(dictid):
  first_letter = ord(string.ascii_lowercase[0])
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

def char2id(char):
  first_letter = ord(string.ascii_lowercase[0])
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0

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

def build_dataset(text,dictionary):
    n = len(text)
    if n%2 != 0:
        n_even = n - 1
        n_odd = n
    else:
        n_even = n
        n_odd = n - 1

    bigram_first = [dictionary[text[i:i+2]] for i in range(0, n_even-2, 2)]
    bigram_second = [dictionary[text[i:i+2]] for i in range(1, n_odd-2, 2)]
    nextchar_first = [char2id(text[i+2]) for i in range(0, n_even-2, 2)]
    nextchar_second = [char2id(text[i+2]) for i in range(1, n_odd-2, 2)]
    bigram = bigram_first + bigram_second
    nextchar = nextchar_first + nextchar_second
    return np.array(bigram),np.array(nextchar)

vocabulary_size = len(string.ascii_lowercase) + 1
url = 'http://mattmahoney.net/dc/'
filename = maybe_download('text8.zip', 31344016)
text = read_data(filename)
# data, count, dictionary, reverse_dictionary = build_dataset(words)
dictionary, reverse_dictionary = build_bigram_dictionary()
bigram,nextchar = build_dataset(text,dictionary)
nextchar = (np.arange(vocabulary_size) == nextchar[:,None]).astype(np.float32)

batch_size = 128
embedding_size = 27 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.
num_steps = 10001
data_index = 0

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):

  # Input data.
  train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, vocabulary_size])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Variables.
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size*vocabulary_size, embedding_size], -1.0, 1.0))
  pickle_file = 'bigramembeddings.pickle'
  try:
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      saved_embeddings = save['embeddings']
      embeddings.assign(saved_embeddings)
      del save  # hint to help gc free up memory

  except Exception as e:
    print('Unable to process data from', pickle_file, ':', e)


  softmax_weights = tf.Variable(
    tf.truncated_normal([embedding_size, vocabulary_size],
                         stddev=1.0 / math.sqrt(embedding_size)))
  softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Model.
  # Look up embeddings for inputs.
  embed = tf.nn.embedding_lookup(embeddings, train_dataset)
  # Compute the softmax loss, using a sample of the negative labels each time.
  logits = tf.matmul(embed,softmax_weights) + softmax_biases
  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, train_labels))

  # Optimizer.
  # Note: The optimizer will optimize the softmax_weights AND the embeddings.
  # This is because the embeddings are defined as a variable quantity and the
  # optimizer's `minimize` method will by default modify all variable quantities
  # that contribute to the tensor it is passed.
  # See docs on `tf.train.Optimizer.minimize()` for more details.
  optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

  # Compute the similarity between minibatch examples and all embeddings.
  # We use the cosine distance:
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):
    offset = (step * batch_size) % (nextchar.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = bigram[np.arange(offset,offset + batch_size)]
    batch_labels = nextchar[np.arange(offset,offset + batch_size)]
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  final_embeddings = normalized_embeddings.eval()

num_points = vocabulary_size*vocabulary_size-1

pickle_file = 'bigramembeddings27.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'embeddings': final_embeddings,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)
