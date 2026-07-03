import pandas as pd
import seaborn as sns
df=pd.read_csv("dataset.csv")
print(df.describe())
print(df.isnull().sum())
sns.heatmap(df.corr(),annot=True)