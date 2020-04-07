#!/usr/bin/env python
# coding: utf-8

# generative model

#使用numpy方便計算高維度陣列與矩陣運算
import numpy as np

"""
step 0

Preparing Data
下載資料

"""

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

x_train, x_mean, x_std = _normalize(x_train, train = True)
x_test, _, _ = _normalize(x_test, train = False, x_mean = x_mean, x_std = x_std)

"""
step 2

計算兩個類別的資料平均與共變異

"""

# 用zip把X_trian和Y_trian包住,這樣就可以同時迭代,如果此筆data y==0 將這筆的x放入X_train_0 
x_train_0 = np.array([x for x, y in zip(x_train, y_train) if y == 0])
x_train_1 = np.array([x for x, y in zip(x_train, y_train) if y == 1])

# 分別計算兩個class(y==0、y==1)分別的mean
# axis = 0,計算各行的mean
mean_0 = np.mean(x_train_0, axis = 0)
mean_1 = np.mean(x_train_1, axis = 0)

# 计算feature的col數
data_dim = x_train.shape[1]

# 計算co_varaince
# 先算兩個分類分別的variance,再依權重加總
cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))

for x in x_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / x_train_0.shape[0]
for x in x_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / x_train_1.shape[0]

cov = (cov_0 * x_train_0.shape[0] + cov_1 * x_train_1.shape[0]) / (x_train_0.shape[0] + x_train_1.shape[0])

"""
step 3

利用mean&covariance
計算weight&bias

"""

# 計算covariance的反矩陣
# matrix可能不可逆,使用SVD的方法求偽逆

u, s, v = np.linalg.svd(cov, full_matrices = False)
inv = np.dot(v.T * 1 / s, u.T)


# 計算weight,bias
w = np.dot(inv, mean_0 - mean_1)
b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))    + np.log(float(x_train_0.shape[0]) / x_train_1.shape[0]) 



"""
step 4

帶入公式預估機率

"""
def _sigmoid(z):
    '''
    sigmoid function can be used to calculate probability
    To avoid overflow, minimum/maximum output value is set
    '''
    # np.clip(a, a_min, a_max)將a限制在a_min和a_max之间，超出範圍的話就是邊界值
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
   
    return _sigmoid(np.matmul(x, w) + b)

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



# Compute accuracy on training set
y_train_predict = 1 - _predict(x_train, w, b)
print('Training accuracy: {}'.format(_accuracy(y_train_predict, y_train)))



"""
step 5

預測testing data
寫入SCV

"""

# Predict testing labels
import csv
y_test_predict = 1 - _predict(x_test, w, b)
with open('predict_generative_model.csv', mode = 'w', newline = '') as f:
    csv_writer = csv.writer(f)
    header = ['id', 'label']
    print(header)
    csv_writer.writerow(header)
    for i in range(y_test_predict.shape[0]):
        row = [str(i), y_test_predict[i]]
        print(row)
        csv_writer.writerow(row)

