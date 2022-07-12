import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())

from sklearn.model_selection import train_test_split

X = df[["glucose", "bloodpressure"]]
y = df[["diabetes"]]

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(X,y,test_size = 0.25, random_state = 42)


from sklearn.naive_bayes import GaussionNB
from sklearn.metrics import accuracy_score
from sklearn.preprosessing import StandardScaler

sc = StandardScaler()

x_train_2 = sc.fit_transform(x_train_2)
x_test_2 = sc.fit_transform(x_test_2)

model_2 = LogisticRegression(random_state = 0)
model_2.fit(x_train_2, y_train_2)

y_pred_2 = model_2.predict(x_test_2)

accuracy = accuracy_score(y_test_2, y_pred_2)
print(accuracy)