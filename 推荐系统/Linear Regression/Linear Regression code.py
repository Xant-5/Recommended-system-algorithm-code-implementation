import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"E:\project\pythonProject\dataset\ml-1m\ratings.dat", sep='::', engine='python',
                   header=None,
                   names=['UserID', 'MovieID', 'Rating', 'Timestamp'], skiprows=1)
features = data.iloc[:, :-1]
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)
X = pd.DataFrame(features_normalized, columns=['UserID', 'MovieID', 'Timestamp'])
Y = data['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.4, random_state=123)
print(X_test)
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算MAE并打印结果
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

# 计算RMSE并打印结果
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Root Mean Squared Error (RMSE):", rmse)
