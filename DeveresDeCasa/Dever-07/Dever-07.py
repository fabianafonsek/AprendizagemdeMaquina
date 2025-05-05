from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Carrega o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Divide os dados: 40% para treino e 60% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4)

# Imprime a quantidade de itens da amostra de teste
print("Quantidade de itens na amostra de teste:", len(X_test))