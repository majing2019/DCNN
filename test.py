#coding=utf8
from model import *
import numpy as np
import time
import os
import tensorflow as tf
import pickle
import util

embed_dim = 32
ws = [8, 5]
top_k = 4
k1 = 19
num_filters = [6, 14]
batch_size = 40
n_epochs = 25
num_hidden = 100
sentence_length = 37
num_class = 6
evaluate_every = 200
checkpoint_every = 200
num_checkpoints = 5
chekpoint_dir = 'runs/1563321186/checkpoints'

f_vocab2id = open('data/vocab2id.pickle','rb')
vocabulary = pickle.load(f_vocab2id)
f_vocab2id.close()
f_id2vocab = open('data/id2vocab.pickle','rb')
vocabulary_inv = pickle.load(f_id2vocab)
f_id2vocab.close()
f_label2id = open('data/label2id.pickle', 'rb')
label2id = pickle.load(f_label2id)
one_hot = np.identity(len(label2id))
f_label2id.close()

sentence = 'What city had a world fair in 1900 ?'
sentence = util.clean_str(sentence)
sentence = sentence.split(" ")
sentences = [sentence]
padded_sentences = []
for i in range(len(sentences)):
    sentence = sentences[i]
    num_padding = sentence_length - len(sentence)
    new_sentence = sentence + ["<PAD/>"] * num_padding
    padded_sentences.append(new_sentence)
sentences = padded_sentences
sentences = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
y = ['enty']
one_hot = np.identity(len(label2id))
label = [one_hot[label2id[label]-1 ] for label in y]
label = np.array(label)

model = DCNN(batch_size, sentence_length, num_filters, embed_dim, top_k, k1, ws, num_hidden, num_class, vocabulary)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    checkpoint_file = tf.train.latest_checkpoint(chekpoint_dir)
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)

    feed_dict = {
        model.sent: sentences,
        model.y: label,
        model.dropout_keep_prob: 1.0
    }

    result = sess.run([model.out], feed_dict=feed_dict)
    print (result)
    