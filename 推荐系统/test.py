import numpy as np
import pandas as pd

data = pd.read_csv(r"D:\pythonProject\Recommended-system-algorithm-code-implementation-main\dataset\ml-1m\ratings.dat", sep='::', engine='python',
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

# 输出二维矩阵
print(ratings)
