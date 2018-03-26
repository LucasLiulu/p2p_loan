#/usr/bin/env python
#coding=utf-8

import tensorflow as tf
import pandas as pd
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
#使用清洗过后的数据，所选特征是经过特征工程处理后的
def data_processing():
    col_new = ['loan_amnt', 'grade', 'open_acc', 'home_ownership_MORTGAGE', 'verification_status_Not Verified',
               'verification_status_Verified', 'purpose_car', 'purpose_debt_consolidation', 'loan_status',
               'purpose_renewable_energy', 'purpose_small_business', 'purpose_vacation', 'term_ 60 months']
    loans = pd.read_csv('/home/ML/p2p_loan_data/loans_2017q2_ml.csv')
    objcol = loans.select_dtypes(include=['O']).columns
    dummy_df = pd.get_dummies(loans[objcol])
    loans = pd.concat([loans, dummy_df], axis=1).drop(objcol, axis=1)
    loans = loans[col_new]
    col = loans.select_dtypes(include=['int64', 'floating']).columns
    col = list(col)
    col.remove('loan_status')
    loans[col] = StandardScaler().fit_transform(loans[col])
    return loans

# 数据处理
loans_df = data_processing()
# 得到训练集和验证集
train_num = int(len(loans_df)*0.7)
feature = list(loans_df.columns)
feature.remove('loan_status')

X_train = loans_df[feature][: train_num]
y_train = loans_df['loan_status'][: train_num]
# 处理不平衡数据
sm = SMOTE(random_state=42)    # 过采样处理的方法
X_train, y_train = sm.fit_sample(X_train, y_train)
'''
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_train, y_train = rus.fit_sample(X_train, y_train)
'''

y_train = pd.DataFrame(pd.Series(y_train)).as_matrix()
X_test = loans_df[feature][train_num:].as_matrix()
y_test = pd.DataFrame(loans_df['loan_status'][train_num:]).as_matrix()

# 将标签转换成one hot
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[0], [1]])
y_train = enc.transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

# 定义节点准备接收数据
xs = tf.placeholder(tf.float32, [None, 12])
ys = tf.placeholder(tf.float32, [None, 2])

# 定义隐藏层和输出层
# 输入值是 xs，有12个特征，在隐藏层有 10 个神经元
w_l1 = tf.Variable(tf.truncated_normal([12, 10], stddev=0.1))
b_l1 = tf.Variable(tf.zeros([1, 10]) + 0.1)
out_l1 = tf.nn.relu(tf.matmul(xs, w_l1) + b_l1)
# 执行dropout
keep_prob = tf.placeholder(tf.float32)
l1_keep = tf.nn.dropout(out_l1, keep_prob)
# 输入值是隐藏层 out_l1，在预测层输出 2 个结果
w_l2 = tf.Variable(tf.truncated_normal([10, 2], stddev=0.1))
b_l2 = tf.Variable(tf.zeros([1, 2]) + 0.1)
y_predict = tf.matmul(l1_keep, w_l2) + b_l2

# 定义交叉熵
cross_entropy = -tf.reduce_sum(ys * tf.log(y_predict))
# 梯度下降最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
predict_correct = tf.equal(tf.argmax(y_predict, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(predict_correct, tf.float32))

# 对所有变量进行初始化
init = tf.initialize_all_variables()
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)

# 迭代 1000 次学习
for i in range(1000):
   # 由 placeholder 定义的运算，这里都要用 feed 传入参数
   sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
   if i % 50 == 0:
       # to see the step improvement
       print('The accuracy is: %.2f' % 1-sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, keep_prob: 1.0}) + 'train times: %d' % i)

    if 1-sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, keep_prob: 1.0}) > bacc:
        bacc = 1-sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, keep_prob: 1.0})

print('The best accuracy is: %.2f' % bacc)
