import matplotlib.pyplot as plt
import numpy as np
from LoadData import load_rating_data, spilt_rating_dat
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF

if __name__ == "__main__":
    file_path = "data/ml-100k/u.data"
    pmf = PMF()  # 建立模型并进行初始化
    pmf.set_params({"num_feat": 10, "alpha": 0.01, "lambda_u": 0.1, "lambda_v": 0.1, "max_epoch": 60, "num_batches": 100,
                    "batch_size": 500})
    ratings = load_rating_data(file_path)  # 获得数据集
    train, test = train_test_split(ratings, test_size=0.2)  # 将数据集拆分为训练集和测试集
    pmf.train(train, test)  # 训练

    # 画出在训练集和测试集上的误差变化
    plt.plot(range(pmf.max_epoch), pmf.rmse_train, marker='.', label='Training Data')
    plt.plot(range(pmf.max_epoch), pmf.rmse_test, marker='o', label='Test Data')
    plt.title('MovieLens')
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
