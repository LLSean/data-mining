#encoding=utf-8
import numpy as np
import pandas as pd
import pickle
import logging
from collections import Counter
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
import codecs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
def onehot_encoder(labels, NUM_CLASSES):
    enc = LabelEncoder()
    labels = enc.fit_transform(labels)
    labels = labels.astype(np.int32)
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size,1), 1)
    concated = tf.concat([indices, labels] , 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0) 
    with tf.Session() as sess:
        return sess.run(onehot_labels)

def load_iris_dataset():
    header = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
    df_iris = pd.read_csv('data/iris.csv', sep=',', names=header)
    df_iris = df_iris[(df_iris['label']!='setosa')]
    labels = onehot_encoder(df_iris['label'], 2)
    cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X_train, X_test, y_train, y_test = train_test_split(df_iris[cols].values, labels, test_size=0.2, random_state=0)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return X_train, y_train, X_test, y_test



if __name__ == '__main__':
    load_iris_dataset()
