#  write your code here 
import pandas as pd

df = pd.read_csv('data/dataset/input.txt')
print(df.groupby(['location'])['height'].apply(lambda x: x.fillna(x.mean()).round(1)).sum())