import pandas as pd
import numpy as np

def ALS(R, K, max_iter=10, reg=0.1):
    m, n = R.shape
    U = np.random.rand(m, K)
    V = np.random.rand(n, K)

    for it in range(max_iter):
        # 更新 V
        for j in range(n):
            V_j = V[j, :]
            R_j = R[:, j]
            mask = ~np.isnan(R_j)
            if sum(mask) == 0:
                continue
            U_masked = U[mask, :]
            R_j_masked = R_j[mask]
            V_j = np.linalg.solve(np.dot(U_masked.T, U_masked) + reg*np.eye(K), np.dot(U_masked.T, R_j_masked))
            V[j, :] = V_j

        # 更新 U
        for i in range(m):
            U_i = U[i, :]
            R_i = R[i, :]
            mask = ~np.isnan(R_i)
            if sum(mask) == 0:
                continue
            V_masked = V[mask, :]
            R_i_masked = R_i[mask]
            U_i = np.linalg.solve(np.dot(V_masked.T, V_masked) + reg*np.eye(K), np.dot(V_masked.T, R_i_masked))
            U[i, :] = U_i

    return U, V

if __name__ == '__main__':
    # 加载数据
    # 读取用户-电影评分数据的前100行
    data = pd.read_csv(
        r"D:\pythonProject\Recommended-system-algorithm-code-implementation-main\dataset\ml-1m\ratings.dat",
        sep='::', engine='python',
        header=None,
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'], skiprows=1
    )

    # 构建用户-电影评分矩阵
    user_movie_matrix = data.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0).values
    print(user_movie_matrix)
    # 预测评分
    U, V = ALS(user_movie_matrix, K=10)
    user_idx = -1
    movie_idx = 1
    rating_pred = np.dot(U[user_idx, :], V.T[:, movie_idx])
    print('预测评分：', round(rating_pred))
