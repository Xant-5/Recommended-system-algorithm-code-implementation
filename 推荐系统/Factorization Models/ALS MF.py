import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

data = pd.read_csv(r"D:\pythonProject\Recommended-system-algorithm-code-implementation-main\dataset\ml-1m\ratings.dat",
                   sep='::', engine='python',
                   header=None,
                   names=['UserID', 'MovieID', 'Rating', 'Timestamp'], skiprows=1)

max_userid = data['UserID'].drop_duplicates().max()
max_movieid = data['MovieID'].drop_duplicates().max()

# 根据评分数据创建二维矩阵
ratings = np.zeros((max_userid, max_movieid))
for row in data.itertuples():
    userid = row[1] - 1  # 用户ID从0开始
    movieid = row[2] - 1  # 电影ID从0开始
    rating = row[3]
    ratings[userid][movieid] = rating
length = ratings.shape[0]  # 行数，即用户数量
width = ratings.shape[1]  # 列数，即电影数量


class ALSMF:
    def __init__(self, n_users, n_items, n_factors=10, alpha=0.01, lmbda=0.1, max_iter=10):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.alpha = alpha
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.user_factors = np.random.normal(scale=1.0 / n_factors, size=(n_users, n_factors))
        self.item_factors = np.random.normal(scale=1.0 / n_factors, size=(n_items, n_factors))

    def fit(self, ratings):
        for _ in range(self.max_iter):
            self.update_user_factors(ratings)
            self.update_item_factors(ratings)

    def update_user_factors(self, ratings):
        confidence = 1 + self.alpha * ratings
        C = np.diag(confidence.sum(axis=1))
        X = self.item_factors.T.dot(C.dot(self.item_factors)) + np.eye(self.n_factors) * self.lmbda
        for u in range(self.n_users):
            y = (ratings[u] * confidence[u]).dot(self.item_factors)
            self.user_factors[u] = spsolve(X, y)

    def update_item_factors(self, ratings):
        confidence = 1 + self.alpha * ratings
        C = np.diag(confidence.sum(axis=0))
        X = self.user_factors.T.dot(C.dot(self.user_factors)) + np.eye(self.n_factors) * self.lmbda
        for i in range(self.n_items):
            y = (ratings[:, i] * confidence[:, i]).dot(self.user_factors)
            self.item_factors[i] = spsolve(X, y)

    def predict(self, user_id, item_id):
        return self.user_factors[user_id].dot(self.item_factors[item_id])


# 创建ALS MF模型，并训练
als = ALSMF(n_users=length, n_items=width, n_factors=10, alpha=0.01, lmbda=0.1, max_iter=10)
als.fit(ratings)

# 预测用户1对物品2的评分
user_id = 1
item_id = 2
predicted_rating = als.predict(user_id, item_id)
print(f"Predicted rating for user {user_id} and item {item_id}: {predicted_rating}")
