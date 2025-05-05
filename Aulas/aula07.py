import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# Carregando o conjunto de dados Iris
iris = datasets.load_iris()

X = iris.data[:, :2]  # Usar apenas as duas primeiras características
y = iris.target

# Criando o classificador KNN
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Definindo os limites do gráfico
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

# Prevendo a classe para cada ponto da malha
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Definindo colormaps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])  # cores do fundo
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])   # cores dos pontos

# Exibindo as regiões de decisão
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plotando os dados de treino
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title("KNN (k = 3) - Classificação das flores Iris")
plt.axis('tight')
plt.show()