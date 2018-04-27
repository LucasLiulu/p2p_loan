#/usr/bin/env python
#coding=utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

learning_rate_base = 0.8    # 基础学习率
learning_rate_decay = 0.99  # 学习率衰减率
regularization_rate = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
train_steps = 30000 
moving_average_decay = 0.99   # 滑动平均衰减率
batch_size = 100
acc = 0.0
dec_rounds = 0
train_times = 0

# 1.训练的数据
def data_processing():
    col_new = ['loan_amnt', 'grade', 'open_acc', 'home_ownership_MORTGAGE','verification_status_Not Verified',
               'verification_status_Verified','purpose_car','purpose_debt_consolidation','loan_status',
               'purpose_renewable_energy', 'purpose_small_business','purpose_vacation', 'term_ 60 months']
    loans = pd.read_csv('E:\\project\\lending_clube\\training_data\\loans_2017q2_ml.csv')
    objcol = loans.select_dtypes(include=['O']).columns
    dummy_df = pd.get_dummies(loans[objcol])
    loans = pd.concat([loans, dummy_df], axis=1).drop(objcol, axis=1)
    loans = loans[col_new]
    col = loans.select_dtypes(include=['int64', 'floating']).columns
    col = list(col)
    col.remove('loan_status')
    loans[col] = StandardScaler().fit_transform(loans[col])
    return loans

loans_df = data_processing()
train_num = int(len(loans_df)*0.7)
feature = list(loans_df.columns)
feature.remove('loan_status')

X_train = loans_df[feature][: train_num]
y_train = loans_df['loan_status'][: train_num]
# 处理不平衡数据
'''
sm = SMOTE(random_state=42)    # 处理过采样的方法
X_train, y_train = sm.fit_sample(X_train, y_train)
'''
rus = RandomUnderSampler(random_state=0)
X_train, y_train = rus.fit_sample(X_train, y_train)
#X_train = X_train.as_matrix()
y_train = pd.DataFrame(pd.Series(y_train)).as_matrix()
X_test = loans_df[feature][train_num:].as_matrix()
y_test = pd.DataFrame(loans_df['loan_status'][train_num:]).as_matrix()

#将标签转换成one hot
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[0], [1]])
y_train = enc.transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()
n_samples = len(y_test)

# 添加层
def add_layer(inputs, weight, biases, avg_class=None, activation_function=None):
    '''
    if inputs is not None:
        print('the inputs value is None!!!')
        return None
    '''
    if avg_class and inputs:
        Wx_plus_b = tf.matmul(inputs, avg_class.average(weight)) + avg_class.average(biases)
    else:
        Wx_plus_b = tf.matmul(inputs, weight) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 2.定义节点准备接收数据
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 12])
ys = tf.placeholder(tf.float32, [None, 2])

# 3.定义神经层：隐藏层和预测层
# 定义滑动平均
global_step = tf.Variable(0, trainable=False)
avg_ema = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
variable_averages_op = avg_ema.apply(tf.trainable_variables())
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
weight1 = tf.Variable(tf.random_normal([12, 28*28], stddev=1, seed=1), name='weight1')
biases1 = tf.Variable(tf.constant(0.1, shape=[28*28]), name='biases1')
l1 = add_layer(xs, weight1, biases1, avg_class=None, activation_function=tf.nn.relu)
# 添加一个卷积层
filter_weight1 = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=1, seed=1))
filter_biases1 = tf.Variable(tf.constant(0.1, shape=[32]))
conv1_in = tf.reshape(l1, [-1, 28, 28, 1])
conv1 = tf.nn.relu(conv2d(conv1_in, filter_weight1) + filter_biases1)
#conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1_in, filter_weight1, strides=[1, 2, 2, 1], padding='SAME'), 
#                                  conv_biases1))
# 添加池化层
pool1 = max_pool(conv1)
#pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
#print('pool1 shape: ', pool1)
filter_weight2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=1, seed=1))
filter_biases2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(conv2d(pool1, filter_weight2) + filter_biases2)
pool2 = max_pool(conv2)

# add output layer 输入值是隐藏层 l1，在预测层输出 1 个one hot结果
weight2 = tf.Variable(tf.random_normal([7*7*64, 1024], stddev=1, seed=1), name='weight2')
biases2 = tf.Variable(tf.constant(0.1, shape=[1024]), name='biases2')
pool2_plat = tf.reshape(pool2, [-1, 7*7*64])
fc_out1 = add_layer(pool2_plat, weight2, biases2, avg_class=None, activation_function=tf.nn.relu)
drop_prob = tf.placeholder(tf.float32)
drop_out1 = tf.nn.dropout(fc_out1, drop_prob)
weight3 = tf.Variable(tf.random_normal([1024, 2], stddev=1, seed=1))
biases3 = tf.Variable(tf.constant(0.1, shape=[2]))
prediction = add_layer(drop_out1, weight3, biases3, avg_class=None, activation_function=None)
# 4.定义 loss 表达式
#print('the ys: ', ys.eval)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=ys))
regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
regularition = regularizer(weight1) + regularizer(weight2) + regularizer(weight3) + regularizer(filter_weight1) + regularizer(filter_weight2)
loss += regularition

# 5.选择 optimizer 使 loss 达到最小
# 使用指数衰减学习率
learning_rate = tf.train.exponential_decay(learning_rate_base, 
                                          global_step, 
                                          n_samples / batch_size,
                                          learning_rate_decay)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_op = tf.group(train_step, variable_averages_op)
cross_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

# important step 对所有变量进行初始化
init = tf.global_variables_initializer()
sess = tf.Session()
with tf.Session() as sess:
    # 上面定义的都没有运算，直到 sess.run 才会开始运算
    sess.run(init)

    for i in range(train_steps):
        start = (i * batch_size) % n_samples
        end = min(start+batch_size, n_samples)
        # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
        #sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
        sess.run(train_step, feed_dict={xs: X_train[start: end], ys: y_train[start: end], drop_prob: 0.5})
        #print ('start: {}, end: {}'.format(start, end))
        #sess.run(train_op, feed_dict={xs: X_train[start: end], ys: y_train[start: end]})
        if i % 500  == 0:
            # to see the step improvement
            test_acc = sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, drop_prob: 1.0}) * 100
            if i and (test_acc > acc):
                acc = test_acc
                train_times = i
            else:
                dec_rounds += 1
            if dec_rounds > 40:
                print('accuracy have not increase in %d rounds, stop training!' % dec_rounds)
                break
            print('The accuracy is: {:.2f}%'.format(test_acc))

    print('the best accuracy is %.2f%%, training times: %d' % (acc, train_times))

