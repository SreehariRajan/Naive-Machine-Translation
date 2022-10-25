from os import getcwd
from utils import get_dict, cosine_similarity
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples
from gensim.models import KeyedVectors
import pdb
import pickle
import string

import time

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import scipy
import sklearn

nltk.download('stopwords')
nltk.download('twitter_samples')

# filePath = f"{getcwd()}/../tmp2/"

# nltk.data.path.append(filePath)

en_embeddings = pickle.load(open("en_embeddings.p", "rb"))
fr_embeddings = pickle.load(open("fr_embeddings.p", "rb"))

en_fr_train = get_dict('en-fr.train.txt')
print("length of training set", len(en_fr_train))

en_fr_test = get_dict('en-fr.test.txt')
print("length of test set", len(en_fr_test))


def get_matrices(en_fr, english_vects, french_vects):
    X_1 = list()
    Y_1 = list()

    english_set = english_vects.keys()
    french_set = french_vects.keys()

    for en_word, fr_word in en_fr.items():
        if fr_word in french_set and en_word in english_set:
            en_vect = english_vects[en_word]

            fr_vect = french_vects[fr_word]

            X_1.append(en_vect)
            Y_1.append(fr_vect)

    X = np.vstack(X_1)
    Y = np.vstack(Y_1)

    return X, Y


X_train, Y_train = get_matrices(en_fr_train, en_embeddings, fr_embeddings)


def compute_loss(X, Y, R):
    m = X.shape[0]

    diff = np.dot(X, R)-Y
    diff_squared = diff**2
    sum_diff_squared = np.sum(diff_squared)

    loss = sum_diff_squared/m

    return loss


def compute_gradient(X, Y, R):
    m = X.shape[0]

    gradient = np.dot(np.transpose(X), np.dot(X, R)-Y)*(2/m)

    return gradient


def train_R(X, Y, train_steps=100, learning_rate=0.0003):

    np.random.seed(129)
    R = np.random.rand(X.shape[1], X.shape[1])

    for i in range(train_steps):
        if i % 10 == 0:
            print(f"loss at iteration {i} is:{compute_loss(X,Y,R):.4f}")

        gradient = compute_gradient(X, Y, R)

        R -= learning_rate*gradient

    return R


R_trained = train_R(X_train, Y_train, train_steps=400, learning_rate=0.8)


def nearest_neighbor(v, candidates, k=1):
    similarity = []

    for row in candidates:
        cos_similarity = cosine_similarity(v, row)

        similarity.append(cos_similarity)

    sorted_ids = np.argsort(similarity)

    k_indeces = sorted_ids[-k:]
    return k_indeces


def test_vocabulary(X, Y, R):
    pred = np.dot(X, R)

    num_correct = 0

    for i in range(len(pred)):
        pred_idx = nearest_neighbor(pred[i], Y)

        if pred_idx == i:
            num_correct += 1

    accuracy = num_correct/len(pred)

    return accuracy


X_val, Y_val = get_matrices(en_fr_test, fr_embeddings, en_embeddings)

acc = test_vocabulary(X_val, Y_val, R_trained)

print(f"Accuracy on test set is {acc}")
