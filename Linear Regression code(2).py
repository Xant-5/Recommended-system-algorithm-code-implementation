from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv(r"E:\project\pythonProject\dataset\ml-1m\ratings.dat", sep='::', engine='python',
                   header=None,
                   names=['UserID', 'MovieID', 'Rating', 'Timestamp'], skiprows=1)
features = data.iloc[:, :-1]
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)
X = pd.DataFrame(features_normalized, columns=['UserID', 'MovieID', 'Timestamp'])
Y = data['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.4, random_state=123)
# 初始化模型参数
theta = np.zeros((X_train.shape[1], 1))
alpha = 0.01
epochs = 10000


# 定义损失函数和梯度函数
def cost_function(X, y, theta):
    m = len(y)
    J = (1 / (2 * m)) * np.sum((np.dot(X, theta) - y) ** 2)
    return J


def gradient_descent(X, y, theta, alpha, epochs):
    m = len(y)
    J_history = []
    # J_history是一个列表，用于存储每次梯度下降迭代后计算得到的代价函数的值（即每次迭代的损失值）。
    # 在梯度下降算法中，我们希望通过迭代来逐步更新模型参数以使代价函数最小化。为了跟踪训练过程中代价函数值的变化情况，我们将每次迭代后的代价函数值添加到J_history列表中。
    for i in range(epochs):
        theta = theta - (alpha / m) * np.dot(X.T, (np.dot(X, theta) - y))
        # 在梯度下降算法中，学习率
        # alpha
        # 确实会直接乘以梯度来计算参数的更新量。而将梯度除以样本数量
        # m，是为了对学习率进行缩放，以便在不同规模的数据集上获得相似的参数更新量。
        #
        # 具体来说，梯度下降算法通过计算损失函数关于参数的梯度来确定参数的更新方向和步长。更新方向是梯度的反方向，也就是使损失函数下降最快的方向。更新步长由学习率
        # alpha
        # 控制，学习率越大，每次更新的步长也就越大。然而，在面对大规模数据集时，每次参数更新的步长可能会过大，从而导致算法无法正常收敛。因此，需要控制步长大小，以确保算法在不同规模的数据集上都能正常运行。
        #
        # 将梯度除以样本数量
        # m，可以将梯度的大小适当缩小。这样一来，在相同的学习率下，每个样本带来的平均梯度就会更小，从而使参数更新的幅度更加合理。同时，这种缩放方法还可以避免梯度下降算法中出现的数值问题，比如梯度爆炸或梯度消失等。
        #
        # 因此，在梯度下降算法中，将学习率
        # alpha
        # 除以样本数量
        # m，是为了控制参数更新的步长大小，以便在不同规模的数据集上获得相似的更新量，并确保算法可以正常收敛。
        J_history.append(cost_function(X, y, theta))

    return theta, J_history


# 训练模型
theta, J_history = gradient_descent(X_train, y_train.values.reshape(-1, 1), theta, alpha, epochs)
# 预测测试集结果
y_pred = np.dot(X_test, theta)

# 计算模型的MAE和RMSE

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('Mean Absolute Error (MAE): ', mae)
print('Root Mean Squared Error (RMSE): ', rmse)
