#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression


import numpy as np

np.random.seed(0)

x_train_fpath = './data/X_train'
y_train_fpath = './data/Y_train'
x_test_fpath  = './data/X_test'
# 將CSV檔轉為numpy array
with open(x_train_fpath, mode = 'r') as f:

    #跳過第一行,因為第一行是feature name
    next(f)

    #strip刪除頭尾指定字符
    #split指定字符分割
    #[1:]跳過第0個元素,從第一行開始,因為col_0代表id
    x_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

with open(y_train_fpath, mode = 'r') as f:
    next(f)
    y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
    
with open(x_test_fpath, mode = 'r') as f:
    next(f)
    x_test  = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    

"""
step 1

Preparing Data
正規化資料

"""


# normalize
def _normalize(x, train = True, specified_column = None, x_mean = None, x_std = None):
    # 因為testing data和training data要用一樣的方式normalize
    # 如果train為True,必須return training data的mean跟std
    # 如果trian為False,必須已知trian出來的mean、std
    #
    # Arguments:
    #   X:要被正規化的data
    #   train:判斷input data是train or test
    #   specified_col:是否有指定哪些col要被正規化,如果是none就全部正規化
    #
    # Outputs:
    #   X:normailize data
    #   X_mean:training data的mean
    #   X_std:training data的std

    if specified_column == None:
        # 所有的col都要被正規化
        specified_column = np.arange(x.shape[1])
        # specified_col->array([0, 1, 2, 3, 4, 5, 6, 7,....,x的col數])
    if train:
        x_mean = np.mean(x[:, specified_column], axis = 0).reshape(1, -1)
        x_std  = np.std(x[:, specified_column], axis = 0).reshape(1, -1)

    x[:, specified_column] = (x[:, specified_column] - x_mean) / (x_std + 1e-8)
    # 避免std過小產生overflow,加上1e-8

    return x, x_mean, x_std

# 切分為訓練集與發展集
def _train_split(x, y, validation_ratio = 0.25):
    '''
    This function splits data into training set and validation set
    '''
    train_size = int(len(x) * (1 - validation_ratio))
    
    #return x,y of training set and validation set  
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:]

# normalize training data and testing data
x_train, x_mean, x_std = _normalize(x_train, train = True)
x_test, _, _ = _normalize(x_test, train = False, x_mean = x_mean, x_std = x_std)

# split training data into training set and validation set
x_training_set, y_training_set, x_validation_set, y_validation_set = _train_split(x_train, y_train, validation_ratio = 0.1)

print('x_training_set : ', x_training_set.shape, '\n', x_training_set)
print('------------------------------------------------------------------------')
print('y_training_set : ', y_training_set.shape, '\n', y_training_set)
print('------------------------------------------------------------------------')
print('x_validation_set : ', x_validation_set.shape, '\n', x_validation_set)
print('------------------------------------------------------------------------')
print('y_validation_set : ', y_validation_set.shape, '\n', y_validation_set)


# some useful functions
def _shuffle(x, y):
    '''
    This function shuffles two equal-length list/array, x and y, together
    '''
    # 打亂原本的順序
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    
    return x[randomize], y[randomize]

def _sigmoid(z):
    '''
    sigmoid function can be used to calculate probability
    To avoid overflow, minimum/maximum output value is set
    '''
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(x, w, b):
    '''
    logistic regression function, parameterized by w and b
    
    Arguements:
        X: input data, shape = [batch_size, data_dimension]
        w: weight vector, shape = [data_dimension, ]
        b: bias, scalar
    output:
        predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    '''
    return _sigmoid(np.dot(x, w) + b)

def _predict(x, w, b):
    '''
    This function returns a truth value prediction for each row of x
    by round function to make 0 or 1
    '''
    # 利用round,四捨五入把機率轉成0或1
    return np.round(_f(x, w, b)).astype(np.int)
    
def _accuracy(y_predict, y_label):
    '''
    This function calculates prediction accuracy
    '''
    # label和預估相減,取絕對值後求mean
    acc = 1 - np.mean(np.abs(y_predict - y_label))
    
    return acc

"""
step 2

判斷function好壞

"""

# loss function(cross_entropy) and gradient
def _cross_entropy_loss(y_predict, y_label):
    '''
    This function computes the cross entropy 
    
    Arguements:
        y_pred: probabilistic predictions, float vector
        Y_label: ground truth labels, bool vector
    Output:
        cross entropy, scalar    
    '''
    # cross_entropy = -Σ(y_head*ln(y)+(1-y_head)*ln(1-y))
    cross_entropy = -(np.dot(y_label, np.log(y_predict)) + np.dot((1 - y_label), np.log(1 - y_predict)))
    
    return cross_entropy

def _gradient(x, y_label, w, b):
    '''
    This function computes the gradient of cross entropy loss with respect to weight w and bias b
    loss function: -Σ (y_head*ln(y)+(1-y_head)*ln(1-y)), 分别對w和b求偏微分，可得
    gradient of w: -Σ（y_head - y)*x
    gradient of b: -Σ（y_head - y)
    '''
    y_predict = _f(x, w, b)
    # 也可以是w_gradient = -np.dot(x.T, y_label - y_predict)
    w_gradient = -np.sum((y_label - y_predict) * x.T, 1)
    b_gradient = -np.sum(y_label - y_predict)
    
    return w_gradient, b_gradient

"""
step 3

adagrad

"""


train_size = x_training_set.shape[0]
validation_size = x_validation_set.shape[0]
dim = x_training_set.shape[1]
# initialize w and b
w = np.zeros(dim)
b= np.zeros(1)

# parameters for training
max_iter = 1000
learning_rate = 1

# save the loss and accuracy
training_set_loss = []
training_set_acc = []
validation_set_loss = []
validation_set_acc = []

w_adagrad = 1e-8
b_adagrad = 1e-8


# training for iterations
for epoch in range(max_iter):
        
    # compute the gradient
    w_gradient, b_gradient = _gradient(x_training_set, y_training_set, w, b)

    # compute the adagrad
    w_adagrad = w_adagrad + np.power(w_gradient, 2)
    b_adagrad = b_adagrad + np.power(b_gradient, 2)

    # gradient descent update 
    # learning rate decay with time
    w = w - learning_rate * w_gradient / np.sqrt(w_adagrad)
    b = b - learning_rate * b_gradient / np.sqrt(b_adagrad)

        
    # one epoch: compute loss and accuracy of training set and validation set
    y_training_predict = _predict(x_training_set, w, b) 
    y_probability = _f(x_training_set, w, b)
    acc = _accuracy(y_training_predict, y_training_set)
    loss = _cross_entropy_loss(y_probability, y_training_set) / train_size # average cross_entropy
    training_set_acc.append(acc)
    training_set_loss.append(loss)
    print('training_set_acc_%d   : %f \t training_set_loss_%d  : %f'%(epoch, acc, epoch, loss))
    
    y_validation_predict = _predict(x_validation_set, w, b)
    y_probability = _f(x_validation_set, w, b)
    acc = _accuracy(y_validation_predict, y_validation_set)
    loss = _cross_entropy_loss(y_probability, y_validation_set) / validation_size # average cross_entropy
    validation_set_acc.append(acc)
    validation_set_loss.append(loss)
    
print('validation_set_acc_%d : %f \t validation_set_loss_%d : %f'%(epoch, acc, epoch, loss))
print()
   



import matplotlib.pyplot as plt
# loss curve
plt.plot(training_set_loss)
plt.plot(validation_set_loss)
plt.title('Loss')
plt.legend(['training_set', 'validation_set'])
plt.savefig('loss_gd.png')
plt.show()

# accuracy curve
plt.plot(training_set_acc)
plt.plot(validation_set_acc)
plt.title('Accuracy')
plt.legend(['training_set', 'validation_set'])
plt.savefig('acc_gd.png')
plt.show()





# mini-batch stochastic-gradient-descent

train_size = x_train.shape[0]
dim = x_train.shape[1]
# initialize w and b
w = np.zeros(dim)
b= np.zeros(1)

# parameters for training
max_iter = 2000
learning_rate = 1

# save the loss and accuracy
train_loss = []
train_acc = []

w_adagrad = 1e-8
b_adagrad = 1e-8


# training for iterations
for epoch in range(max_iter):
        
    # compute the gradient
    w_gradient, b_gradient = _gradient(x_train, y_train, w, b)

    # compute the adagrad
    w_adagrad = w_adagrad + np.power(w_gradient, 2)
    b_adagrad = b_adagrad + np.power(b_gradient, 2)

    # gradient descent update 
    # learning rate decay with time
    w = w - learning_rate * w_gradient / np.sqrt(w_adagrad)
    b = b - learning_rate * b_gradient / np.sqrt(b_adagrad)
    
    y_train_predict = _predict(x_train, w, b) 
    y_probability = _f(x_train, w, b) 
    acc = _accuracy(y_train_predict, y_train)
    loss = _cross_entropy_loss(y_probability, y_train) / train_size # average cross_entropy
    train_acc.append(acc)
    train_loss.append(loss)
    print('train_acc_%d   : %f \t train_loss_%d  : %f'%(epoch, acc, epoch, loss))
        




# loss curve
plt.plot(train_loss)
plt.title('Loss')
plt.legend(['train'])
# plt.show()

# accuracy curve
plt.plot(train_acc)
plt.title('Accuracy')
plt.legend(['train'])
# plt.show()

np.save('weight_adagrad_gd.npy', w)
np.save('bias_adagrad_gd.npy', b)



# predict testing data
import csv
# w = np.load('weight_adagrad.npy')
# b = np.load('bias_adagrad.npy')
y_test_predict = _predict(x_test, w, b)
print(y_test_predict, y_test_predict.shape)

with open('predict_adagrad_gd.csv', mode = 'w', newline = '') as f:
    csv_writer = csv.writer(f)
    header = ['id', 'label']
    print(header)
    csv_writer.writerow(header)
    for i in range(y_test_predict.shape[0]):
        row = [str(i), y_test_predict[i]]
        csv_writer.writerow(row)
        print(row)
