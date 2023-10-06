import pandas as pd

data = pd.read_csv(
    r"D:\pythonProject\Recommended-system-algorithm-code-implementation-main\dataset\archive\dataset_TSMC2014_NYC.csv",
    sep='::', engine='python',
    header=None,
    names=['UserID', 'MovieID', 'Rating', 'Timestamp'], skiprows=1)