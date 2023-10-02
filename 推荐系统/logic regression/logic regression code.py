import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(r"E:\project\pythonProject\dataset\ml-1m\ratings.dat", sep='::', engine='python',
                   header=None,
                   names=['UserID', 'MovieID', 'Rating', 'Timestamp'], skiprows=1)

for i in range(len(data['Rating'])):
    if data.iloc[i, 2] <= 2:
        data.iloc[i, 2] = 0
    else:
        data.iloc[i, 2] = 1
print(data)
features = data.iloc[:, :-1]
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)
X = pd.DataFrame(features_normalized, columns=['UserID', 'MovieID', 'Timestamp'])
Y = data['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.4, random_state=123)
