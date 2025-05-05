import numpy as np
from matplotlib import pyplot as plt
from sklearn import neighbors, datasets

iris = datasets.load_iris()

X = iris.data[:, :2] #Pegar apenas as duas primeiras características

y = iris.target

knn = neighbors.KNeighborsClassifier(n_neighbors=3)

knn.fit(X, y)

x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1

y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

print (np.c_[xx.ravel(), yy.ravel()])

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

print (Z)