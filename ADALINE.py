class AdalineGD(object):
        """ADAptive LInear NEuron classifier

        Parameters
        -----------
        eta : float
            Learning rate (between 0.0 and 1.0)
        n_iter : int
            Passes over the training dataset
        
        Attributes
        -----------
        w_ : 1d_array
            Weights after fitting
        errors_ : list
            Number of misclassifications in every epoch
        
        """
        def __init__(self, eta=0.01, n_iter=50):
            self.eta = eta
            self.n_iter = n_iter
        
        def fit(self, X, y):
            """Fit training data

            Parameters
            ----------
            X : {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples
                is the number of samples and
                n_features is the number of features
            y : array-like, shape = [n_samples]
                Target values

            Returns
            -------
            self : object

            """
            self.w_ = np.zeros(1 + X.shape[1])
            self.cost_ = []

            for i in range(self.n_iter):
                output = self.net_input(X)
                errors = (y - output)
                self.w_[1:] += self.eta * X.T.dot(errors)
                self.w_[0] += self.eta * errors.sum()
                cost = (errors**2) / 2.0
                self.cost_.append(cost)
            return self

        def net_input(self, X):
            """Calculate net input"""
            return np.dot(X, self.w_[1:]) + self.w_[0]

        def activation(self, X):
            """Compute linear activation"""
            return self.net_input(X)

        def predict(self, X):
            """return class label after unit step"""
            return np.where(self.activation(X) >= 0.0, 1, -1)

from matplotlib.colors import ListedColormap
import pandas as pd 
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np 

def plot_decision_generator(X, y, classifier, resolution=0.02):
    #setup marker generator and color map
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class samples
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y = X[y == c1, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=c1)

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()



"""take the label of the plant from the data"""
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
"""take first 200 int elements, just property 0 and 2"""
X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="versicolor")
plt.xlabel("petal length")
plt.ylabel("sepal length")
plt.legend(loc="upper left")
"""printing data"""
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
adal = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(adal.cost_) + 1), np.log10(adal.cost_), marker="o")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("log(Sum-squared-error)")
ax[0].set_title("Adaline - learning rate 0.01")
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker="o")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Sum-squared-error")
ax[1].set_title("Adaline - learning rate 0.0001")
plt.show()

""" data standardization"""
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, clf=ada)
plt.title("Adaline - Gradient decent")
plt.xlabel("Sepal length [standardized]")
plt.ylabel("petal length [standardized]")
plt.legend(loc="upper left")
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Sum-squared-error")
plt.show()

