import pandas as pd 
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
"""
df_wine.columns = ["Class label", "Alcohol",
                    "Malic acid", "Ash",
                    "Alcalinity of ash", "Magnesium",
                    "Total phenols", "Flavanoids",
                    "Nonflavanoid phenols", 
                    "Proanthocyanis",
                    "Color intensity", "Hue",
                    "OD280/OD315 of diluited wines",
                    "Proline"]
"""

#retrieve data values from the dataset using iloc
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
#split data into test and train arrays
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
#standardize data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
#perform PCA
pca = PCA(n_components=2)
pca.fit(X_train_std)  
x_train_pca = pca.transform(X_train_std)
#data is mirrored, invert
x_train_pca = x_train_pca * -1
#plot PCA
colors = ["r", "b", "g"]
markers = ["s", "x", "o"]
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(x_train_pca[y_train==l, 0], x_train_pca[y_train==l, 1], c=c, marker=m, label=l)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc="lower left")
plt.show()