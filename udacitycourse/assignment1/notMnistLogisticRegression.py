# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sklearn import datasets

pickle_file = 'notMNIST.pickle'
try:
  f = open(pickle_file, 'rb')
  output = pickle.load(f)
except Exception as e:
  print('Unable to load data from', pickle_file, ':', e)
  raise

train_dataset = output['train_dataset']
train_dataset = train_dataset.reshape(train_dataset.shape[0],train_dataset.shape[1]*train_dataset.shape[2])
train_labels = output['train_labels']
valid_dataset = output['valid_dataset']
valid_dataset = valid_dataset.reshape(valid_dataset.shape[0],valid_dataset.shape[1]*valid_dataset.shape[2])
valid_labels = output['valid_labels']
test_dataset = output['test_dataset']
test_dataset = test_dataset.reshape(test_dataset.shape[0],test_dataset.shape[1]*test_dataset.shape[2])
test_labels = output['test_labels']

clf0 = LogisticRegression(multi_class='ovr')
clf0.fit(train_dataset[0:1000,:],train_labels[0:1000])
score = clf0.score(valid_dataset,valid_labels)
print(score)

clf1 = LogisticRegression(multi_class='multinomial',solver="sag", max_iter=1000)
clf1.fit(train_dataset[0:1000,:],train_labels[0:1000])
score = clf1.score(valid_dataset,valid_labels)
print(score)

clf2 = LogisticRegression(penalty="l1")
clf2.fit(train_dataset[0:1000,:],train_labels[0:1000])
score = clf2.score(valid_dataset,valid_labels)
print(score)
