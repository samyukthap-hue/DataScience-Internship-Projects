import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("dataset.csv")
print(df.describe())
df.groupby("CustomerSegment")["Sales"].mean().plot(kind="bar")
plt.show()