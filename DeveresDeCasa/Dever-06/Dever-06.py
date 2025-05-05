from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# 1. Carregar os dados
iris = load_iris()
X = iris.data  # características
y = iris.target  # rótulos
nomes_especies = iris.target_names  # nomes das espécies

# 2. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Treinar o modelo nos dados de treino
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)
print("Modelo treinado com sucesso!")

# 4. Entrada do usuário
try:
    print("\nDigite os valores das medidas da flor:")
    comprimento_sepala = float(input("Comprimento da sépala (cm): "))
    largura_sepala = float(input("Largura da sépala (cm): "))
    comprimento_petala = float(input("Comprimento da pétala (cm): "))
    largura_petala = float(input("Largura da pétala (cm): "))

    # 5. Previsão
    entrada = [[comprimento_sepala, largura_sepala, comprimento_petala, largura_petala]]
    predicao = modelo.predict(entrada)

    # 6. Imprimir o nome da flor
    especie_predita = nomes_especies[predicao[0]]
    print(f"\nA flor é da espécie: {especie_predita}")

    # 7. Verificando taxa de acerto
    acertos = metrics.accuracy_score(y_test, modelo.predict(X_test))
    print(f"\nAcurácia do modelo nos dados de teste: {acertos:.2f}")

except ValueError:
    print("Por favor, digite apenas números válidos para as medidas.")
