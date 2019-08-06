import numpy as np
import pandas as pd 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.tail()
#retrieve data values from the dataset using iloc
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
#split data into test and train arrays
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
#create pipeline
pipe_lr = Pipeline([("scl", StandardScaler()),
                    ("pca", PCA(n_components=2),
                    ("clf", LogisticRegression(random_state=1)))])
tot_splits = 10
scores=[]

kfold = StratifiedKFold(n_splits=tot_splits, random_state=1)
for train, test in kfold.split(X_train, y_train):
    pipe_lr.fit(X_train[train], y_train[train])
    scores.append(pipe_lr.score(X_train[test], y_train[test]))

plt.plot(np.arange(1, tot_splits, 1), scores, marker="x")
plt.ylabel("Split")
plt.xlabel("Accuracy")
plt.grid()
plt.show()

