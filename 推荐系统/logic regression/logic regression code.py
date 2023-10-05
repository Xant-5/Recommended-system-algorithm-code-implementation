import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def calculate_recall(y_true, y_pred):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # 计算召回率（Recall）
    recall = tp / (tp + fn)
    return recall


data = pd.read_csv(r"D:\pythonProject\Recommended-system-algorithm-code-implementation-main\dataset\ml-1m\ratings.dat",
                   sep='::', engine='python',
                   header=None,
                   names=['UserID', 'MovieID', 'Rating', 'Timestamp'], skiprows=1)
data['Rating'] = data['Rating'].apply(lambda x: 1 if x >= 3 else 0)
# 数据预处理
scaler = MinMaxScaler()
data['Timestamp'] = scaler.fit_transform(data[['Timestamp']])

train_set = []
test_set = []

current_user_id = None
for item in data.values:
    user_id = item[0]
    if user_id != current_user_id:
        # 将当前用户ID的数据随机划分为训练集和测试集
        random.shuffle(train_set)
        random.shuffle(test_set)
        train_set.extend(train_set)
        test_set.extend(test_set)
        train_set.clear()
        test_set.clear()
        current_user_id = user_id

    # 将数据添加到相应的集合中
    if random.random() < 0.8:
        train_set.append(item)
    else:
        test_set.append(item)

# 最后一批用户ID相同的数据也要进行划分
random.shuffle(train_set)
random.shuffle(test_set)
train_set.extend(train_set)
test_set = [[int(x) for x in inner_lst] for inner_lst in test_set]
train_set = [[int(x) for x in inner_lst] for inner_lst in train_set]
X_train = []
y_train = []
for item in train_set:
    user_id = item[0]
    movie_id = item[1]
    # 将用户ID和电影ID作为输入，将评分作为输出
    X_train.append([user_id, movie_id])
    y_train.append(item[2])

X_test = []
y_test = []
for item in test_set:
    user_id = item[0]
    movie_id = item[1]
    X_test.append([user_id, movie_id])
    y_test.append(item[2])

# 创建逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
recall = calculate_recall(y_test, y_pred)
print("召回率:", recall)
print("准确率:", accuracy)
