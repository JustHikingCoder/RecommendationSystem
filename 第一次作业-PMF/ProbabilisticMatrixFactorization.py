# -*- coding: utf-8 -*-
import numpy as np


class PMF(object):
    def __init__(self, num_feat=10, alpha=0.1, lambda_u=0.1, lambda_v=0.1, max_epoch=20, num_batches=10, batch_size=1000):
        self.num_feat = num_feat  # 潜在特征的数量
        self.alpha = alpha  # 特征向量更新时所采用的学习率
        self.lambda_u = lambda_u  #
        self.lambda_v = lambda_v
        self.max_epoch = max_epoch  # 大循环总共进行max_epoch轮
        self.num_batches = num_batches  # 每一轮进行num_batches批更新
        self.batch_size = batch_size  # 在每一轮批量更新中，一批的样本数

        self.w_movie = None  # 电影特征向量
        self.w_User = None  # 用户特征向量

        self.rmse_train = []
        self.rmse_test = []

    # 使用训练集进行训练，采用批量梯度下降方法，每一轮迭代之后观察在训练集和测试机上的误差
    def train(self, train_vec, test_vec):

        pairs_train = train_vec.shape[0]  # 训练集中的数据条数
        pairs_test = test_vec.shape[0]  # 测试集中的数据条数

        # 计算用户总数和电影总数
        num_user = int(max(np.amax(train_vec[:, 0]), np.amax(test_vec[:, 0]))) + 1  # 第0列，user总数
        num_movie = int(max(np.amax(train_vec[:, 1]), np.amax(test_vec[:, 1]))) + 1  # 第1列，movie总数

        incremental = False  # 增量
        if ((not incremental) or (self.w_movie is None)):
            # initialize
            self.epoch = 0
            self.w_movie = 0.1 * np.random.randn(num_movie, self.num_feat)  # 以0为均值，以0.1为标准差的正态分布 电影浅因子矩阵 M x D 正态分布矩阵
            self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  # 以0为均值，以0.1为标准差的正态分布 用户浅因子矩阵 N x D 正态分布矩阵

        while self.epoch < self.max_epoch:  # 进行迭代
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])  # 根据记录数创建等差array
            np.random.shuffle(shuffled_order)  # 将shuffled中的元素打乱

            # 按批量更新
            for batch in range(self.num_batches):

                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  # 本次迭代要使用的，在shuffled_order中的索引下标

                batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')  # 一维向量，内容为用户id
                batch_movieID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')  # 一维向量，内容为电影id

                # 计算目标函数
                pred_out = np.sum(np.multiply(self.w_User[batch_UserID, :],
                                              self.w_movie[batch_movieID, :]), axis=1)
                # 将用户浅因子向量和商品浅因子向量相乘，再将元素相加，用以模拟矩阵乘法，得到U*V算出的r(i,j)

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2]  # 预测值与观测值的差

                # 计算梯度
                Ix_User = np.multiply(rawErr[:, np.newaxis], self.w_movie[batch_movieID, :]) \
                       + self.lambda_u * (self.w_User[batch_UserID, :])    # batch_size*D维的矩阵
                Ix_movie = np.multiply(rawErr[:, np.newaxis], self.w_User[batch_UserID, :]) \
                       + self.lambda_v * (self.w_movie[batch_movieID, :])

                g_movie = np.zeros((num_movie, self.num_feat))
                g_User = np.zeros((num_user, self.num_feat))

                # 计算对于同一个用户或者商品的梯度和，然后统一进行更新
                for i in range(self.batch_size):
                    g_movie[batch_movieID[i], :] += Ix_movie[i, :]
                    g_User[batch_UserID[i], :] += Ix_User[i, :]
                self.w_movie = self.w_movie - self.alpha * g_movie
                self.w_User = self.w_User - self.alpha * g_User

                # 计算训练集误差
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(train_vec[:, 0], dtype='int32'), :],
                                                  self.w_movie[np.array(train_vec[:, 1], dtype='int32'), :]), axis=1)
                    rawErr = pred_out - train_vec[:, 2]
                    obj = np.linalg.norm(rawErr) ** 2 \
                          + 0.5 * self.lambda_u * np.linalg.norm(self.w_User) ** 2 \
                          + 0.5 * self.lambda_v * np.linalg.norm(self.w_movie) ** 2

                    self.rmse_train.append(np.sqrt(obj / pairs_train))

                # 计算测试集误差
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(test_vec[:, 0], dtype='int32'), :],
                                                  self.w_movie[np.array(test_vec[:, 1], dtype='int32'), :]), axis=1)
                    rawErr = pred_out - test_vec[:, 2]
                    self.rmse_test.append(np.linalg.norm(rawErr) / np.sqrt(pairs_test))

                    # Print info
                    print('Training RMSE: %f, Test RMSE %f' % (self.rmse_train[-1], self.rmse_test[-1]))

    def predict(self, invID):
        return np.dot(self.w_movie, self.w_User[int(invID), :])

    # 设置参数
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.alpha = parameters.get("alpha", 0.1)
            self.lambda_u = parameters.get("lambda_u", 0.1)
            self.lambda_v = parameters.get("lambda_v", 0.1)
            self.max_epoch = parameters.get("max_epoch", 60)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)
