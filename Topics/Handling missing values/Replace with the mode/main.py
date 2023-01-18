#  write your code here
import pandas as pd

df = pd.read_csv('data/dataset/input.txt')
df.location.fillna(df.location.mode()[0], inplace=True)
print(df.head())
