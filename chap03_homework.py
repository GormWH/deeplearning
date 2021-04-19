import os
import sys

import numpy as np
import pandas as pd

sys.modules['tensorflow'] = None

def load_fashionmnist():
    # 学習データ
    x_train = np.load('drive/My Drive/Colab Notebooks/DLBasics2021_colab/Lecture_20210415/data/x_train.npy')
    y_train = np.load('drive/My Drive/Colab Notebooks/DLBasics2021_colab/Lecture_20210415/data/y_train.npy')
    
    # テストデータ
    x_test = np.load('drive/My Drive/Colab Notebooks/DLBasics2021_colab/Lecture_20210415/data/x_test.npy')
    
    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    y_train = np.eye(10)[y_train.astype('int32')]
    x_test = x_test.reshape(-1, 784).astype('float32') / 255
    
    return x_train, y_train, x_test

"""### ソフトマックス回帰の実装"""

# logの中身が0になるのを防ぐ
def np_log(x):
  return np.log(np.clip(a=x, a_min=1e-10, a_max=1e+10))

x_train, y_train, x_test = load_fashionmnist()
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def softmax(x):
  # WRITE ME
  x -= x.max(axis=1, keepdims=True)
  x_exp = np.exp(x)
  return x_exp / np.sum(x_exp, axis = 1, keepdims=True)

# weights
W = np.random.uniform(low=-0.08, high=0.08, size=(784, 10)).astype('float32') # WRITE ME
b = np.zeros(shape=(10,)).astype('float32') # WRITE ME

# 学習データと検証データに分割
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

def train(x, y, eps=1.2):
  # WRITE ME
  global W,b

  batch_size = x.shape[0]

  # 予測
  y_hat = softmax(np.matmul(x,W) + b) # shpae: (batch_size, output)

  # 目的関数の評価
  cost = (-y*np_log(y_hat)).sum(axis=1).mean()
  delta = y_hat - y

  # パラメータの更新
  dW = np.matmul(x.T, delta) / batch_size # shape: (入力の次元数, 出力の次元数)
  db = np.matmul(np.ones(shape=(batch_size,)), delta) / batch_size # shape: (出力の次元数,)
  W -= eps * dW
  b -= eps * db

  return cost

def valid(x, y):
  # WRITE ME
  y_hat = softmax(np.matmul(x, W) + b)
  cost = (- y * np_log(y_hat)).sum(axis=1).mean()

  return cost, y_hat

eps = 1.0 #更新の感度設定
for epoch in range(300):
  # オンライン学習
  # WRITE ME
  x_train, y_train = shuffle(x_train, y_train)
  cost = train(x_train, y_train, 0.1*eps + 0.9*eps*(1-epoch/300)) #更新感度は'1'から'0.1'に単調減少
  cost, y_pred = valid(x_valid, y_valid)
  if epoch % 10 == 9 or epoch == 0:
    print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
            epoch + 1,
            cost,
            accuracy_score(y_valid.argmax(axis=1), y_pred.argmax(axis=1))
        ))

y_pred = softmax(np.matmul(x_test, W) + b).argmax(axis=1)#.reshape(x_test.shape[0],)# WRITE ME

submission = pd.Series(y_pred, name='label')
submission.to_csv('drive/My Drive/Colab Notebooks/DLBasics2021_colab/Lecture_20210415/submission_pred.csv', header=True, index_label='id')