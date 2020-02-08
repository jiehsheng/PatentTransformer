# measure by Universal Sentence Encoder (Google)

# the following code is based on:
# https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb

# Install TF-Hub.

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import sys
import csv
import argparse

#tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(
  description='Measure by Universal Sentence Encoder',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_file', default='/content/span_pairs.tsv', type=str, help='input file', required=False)

parser.add_argument('--output_file', default='', type=str, help='output file', required=False)

parser.add_argument('--use_version', default='4', type=str, help='output file', required=False)

args, unknown = parser.parse_known_args()
if len(unknown) > 0:
  print('unknown: %s' % unknown)
  sys.exit(1)

def main():
  # Import the Universal Sentence Encoder's TF Hub module
  #url_v2 = "https://tfhub.dev/google/universal-sentence-encoder/2"
  #url_v3 = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
  #https://tfhub.dev/google/universal-sentence-encoder/4

  if args.use_version == '4':
    hub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    #embed = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/4')
  elif args.use_version == 'L5':
    hub_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    #embed = hub.Module('https://tfhub.dev/google/universal-sentence-encoder-large/5')
  else:
    print('unknown: %s' % args.use_version)
    sys.exit(1)

  embed = hub.load(hub_url)
  print('using: %s' % hub_url)

  threshold = 0.5
  relevant = 0
  irrelevant = 0
  sim_avg = 0
  sim_total = 0
  with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    messages = tf.placeholder(dtype=tf.string, shape=[None])
    output = embed(messages)

    print('input_file: %s' % args.input_file)
    if args.output_file != '':
      print('output_file: %s' % args.output_file)
      f_out = tf.gfile.Open(args.output_file, "w")
    else:
      f_out = None
    with tf.gfile.Open(args.input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t")
      count = 1
      for i, line in enumerate(reader):
        if i == 0:
          continue
        s1 = line[3].replace('\n', ' ').strip()
        s2 = line[4].replace('\n', ' ').strip()
        if len(s1) == 0 or len(s2) == 0:
          continue

        embed1 = session.run(output, feed_dict={messages: [s1]})
        embed2 = session.run(output, feed_dict={messages: [s2]})
        #embed1 = session.run(embed([s1]))
        #embed2 = session.run(embed([s2]))
        sim_matrix  = np.inner(embed1, embed2)
        v = sim_matrix[0][0]
        sim_total += v
        msg = '[ %s ] similarity = %s [ %s ][ %s ]' % (count, v, s1, s2)
        print(msg)
        if f_out is not None:
          f_out.write('%s\n' % msg) 
        if v >= threshold:
          relevant += 1
        else:
          irrelevant += 1
        count += 1
      '''  
      msg = '(similar) count = %s, ratio = %s' % (relevant, float(relevant)/(relevant+irrelevant))
      print(msg)
      if f_out is not None:
        f_out.write('%s\n' % msg) 
      msg = '(not similar) count = %s, ratio = %s' % (irrelevant, float(irrelevant)/(relevant+irrelevant))
      print(msg)
      if f_out is not None:
        f_out.write('%s\n' % msg) 
      '''
      sim_avg = sim_total / count
      msg = 'average similarity = %s (count=%s)' % (sim_avg, count)
      print(msg)
      if f_out is not None:
        f_out.write('%s\n' % msg) 
    if f_out is not None:
      f_out.close()

if __name__ == '__main__':
  main()