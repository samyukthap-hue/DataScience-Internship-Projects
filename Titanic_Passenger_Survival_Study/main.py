import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv("dataset.csv")
print(df.info())
sns.countplot(x="Survived",data=df)
plt.show()
X=df[["Age","Fare"]]
y=df["Survived"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=RandomForestClassifier()
model.fit(X_train,y_train)
p=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,p))