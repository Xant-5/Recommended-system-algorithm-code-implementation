import pandas as pd
from datetime import datetime

data = pd.read_csv(
    r"D:\pythonProject\Recommended-system-algorithm-code-implementation-main\dataset\archive\dataset_TSMC2014_NYC.csv",
    sep=',', engine='python',
    header=None,
    usecols=[0, 1, 3, 7],
    names=['userId', 'venueId', 'venueCategory', 'utcTimestamp'],
    skiprows=1
)
data['utcTimestamp'] = data['utcTimestamp'].apply(lambda x: datetime.strptime(x, '%a %b %d %H:%M:%S %z %Y'))

# 根据小时数将数据分类
data['timeOfDay'] = pd.cut(data['utcTimestamp'].dt.hour,
                           bins=[0, 12, 18, 24],
                           labels=[1, 2, 3],  # 上午用1表示，下午用2表示，晚上用3表示
                           include_lowest=True)

# 打印结果
print(data)
